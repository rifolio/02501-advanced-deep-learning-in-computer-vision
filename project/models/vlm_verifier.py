"""
VLM-as-Verifier: Uses Vision-Language Model to verify Grounding DINO detections.

For each DINO detection (cropped region):
1. Show cropped detection + support examples to VLM
2. Ask: "Does this detection match the target object?"
3. Extract confidence score from VLM response
4. Filter false positives based on confidence threshold
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from PIL import Image

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)


class VerificationResponse:
    """Structured response from VLM verification."""

    def __init__(
        self,
        decision: str,  # "YES" or "NO"
        confidence: float,  # 0.0 to 1.0
        reasoning: str = "",
        is_valid: Optional[bool] = None,
    ):
        self.decision = decision.upper()  # Normalize to YES/NO
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.reasoning = reasoning
        self.is_valid = (
            is_valid
            if is_valid is not None
            else self.decision == "YES"
        )  # Default: YES → valid

    def __repr__(self):
        return (
            f"VerificationResponse(decision={self.decision}, "
            f"confidence={self.confidence:.2f}, is_valid={self.is_valid})"
        )


class VLMVerifier:
    """
    Verification module using VLM to filter false positives.

    Pipeline:
    1. For each DINO detection, crop the bbox region
    2. Build verification prompt (detection + support examples)
    3. VLM answers: "Does this match the target?"
    4. Extract confidence → filter based on threshold
    """

    def __init__(self, vlm: BaseVLM, device: str = "cuda"):
        """
        Initialize VLM verifier.

        Args:
            vlm: Vision-Language Model instance (e.g., Qwen2.5-VL)
            device: Device for computation ('cuda' or 'cpu')
        """
        self.vlm = vlm
        self.device = device
        self.model_name = f"VLMVerifier-{vlm.model_name}"
        logger.info(f"Initialized {self.model_name}")

    def verify_detection(
        self,
        cropped_detection: Image.Image,
        support_images: list[Image.Image],
        class_name: str,
    ) -> VerificationResponse:
        """
        Verify if a cropped detection matches the target object.

        Args:
            cropped_detection: PIL Image of cropped bbox region from query image
            support_images: List of PIL Images showing target object (from support set)
            class_name: Target class name (e.g., "dog", "cat")

        Returns:
            VerificationResponse with decision and confidence score
        """
        if not support_images:
            logger.warning("No support images provided for verification")
            return VerificationResponse("UNKNOWN", 0.5, "No support images available")

        try:
            # Build verification prompt
            prompt_bundle = self._build_verification_prompt(
                cropped_detection, support_images, class_name
            )

            images = prompt_bundle["images"]
            prompt_text = prompt_bundle["text"]

            # Get VLM response
            response_text = self._query_vlm(images, prompt_text)

            # Parse response
            verification_result = self._parse_verification_response(
                response_text, class_name
            )

            logger.debug(
                f"Verification for {class_name}: {verification_result.decision} "
                f"(confidence={verification_result.confidence:.2f})"
            )

            return verification_result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResponse("ERROR", 0.5, f"Verification error: {str(e)}")

    def _build_verification_prompt(
        self,
        cropped_detection: Image.Image,
        support_images: list[Image.Image],
        class_name: str,
    ) -> dict:
        """
        Build verification prompt with support examples + detected region.

        Returns:
            Dict with 'images' (list of PIL Images) and 'text' (prompt string)
        """
        # Combine support images with the detected region
        images = support_images + [cropped_detection]

        # Build text prompt
        num_supports = len(support_images)
        text = (
            f"You are a verification assistant for object detection.\n\n"
            f"The first {num_supports} image(s) show example(s) of a {class_name}.\n"
            f"The last image shows a detected region from a query image.\n\n"
            f"Question: Does the detected region (last image) show a {class_name}?\n"
            f"Answer with exactly this format:\n"
            f'ANSWER: YES or NO\nCONFIDENCE: [0-100]\nREASONING: [brief explanation]\n\n'
            f"Be strict: only answer YES if the region clearly shows a {class_name}."
        )

        return {"images": images, "text": text}

    def _query_vlm(self, images: list[Image.Image], prompt_text: str) -> str:
        """
        Query VLM with images and prompt.

        Args:
            images: List of PIL Images
            prompt_text: Prompt string

        Returns:
            VLM response text
        """
        # Determine image dimensions from first image
        if not images:
            raise ValueError("No images provided")

        img_width, img_height = images[0].size

        # Use VLM's few-shot prediction interface
        # (assuming VLM can handle multi-image prompts via predict_few_shot)
        try:
            # Try predict_few_shot if available
            if hasattr(self.vlm, "predict_few_shot"):
                response_text = self.vlm.predict_few_shot(
                    query_image=images[-1],  # Last image is the query
                    support_images=images[:-1],  # Remaining are supports
                    prompt_text=prompt_text,
                    img_width=img_width,
                    img_height=img_height,
                )
                if isinstance(response_text, list):
                    # Some models return detections, not text
                    return str(response_text)
                return response_text

        except NotImplementedError:
            pass

        # Fallback: use direct predict (may not work for multi-image)
        logger.warning(
            f"VLM {self.vlm.model_name} does not support predict_few_shot, "
            "verification may be unreliable"
        )
        return ""

    def _parse_verification_response(
        self, response_text: str, class_name: str
    ) -> VerificationResponse:
        """
        Parse VLM response to extract decision and confidence.

        Expected format:
            ANSWER: YES
            CONFIDENCE: 85
            REASONING: The detected region clearly shows a [class_name]

        Args:
            response_text: Raw VLM response
            class_name: Target class name (for logging)

        Returns:
            VerificationResponse with parsed decision and confidence
        """
        if not response_text or not response_text.strip():
            logger.warning(f"Empty VLM response for {class_name} verification")
            return VerificationResponse("UNKNOWN", 0.5, "Empty response")

        response_lower = response_text.lower()

        # Extract decision (YES/NO)
        decision = "UNKNOWN"
        if "yes" in response_lower:
            decision = "YES"
        elif "no" in response_lower:
            decision = "NO"

        # Extract confidence score
        confidence = 0.5  # Default
        confidence_match = re.search(
            r"confidence[:\s]+(\d+(?:\.\d+)?)", response_lower
        )
        if confidence_match:
            conf_value = float(confidence_match.group(1))
            # Normalize to [0, 1] if in [0, 100]
            if conf_value > 1.0:
                confidence = conf_value / 100.0
            else:
                confidence = conf_value
        else:
            # Try to infer from response text
            if decision == "YES":
                confidence = 0.7  # Default positive confidence
            elif decision == "NO":
                confidence = 0.3  # Default negative confidence

        # Extract reasoning
        reasoning = ""
        reasoning_match = re.search(
            r"reasoning[:\s]+(.+?)(?:\n|$)", response_text, re.IGNORECASE
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        logger.debug(
            f"Parsed verification response for {class_name}: "
            f"decision={decision}, confidence={confidence:.2f}, reasoning={reasoning[:50]}"
        )

        return VerificationResponse(
            decision=decision, confidence=confidence, reasoning=reasoning
        )

    def batch_verify_detections(
        self,
        query_image: Image.Image,
        detections: list[list[float]],
        support_images: list[Image.Image],
        class_name: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[list[dict], dict]:
        """
        Verify multiple detections from a single image.

        Args:
            query_image: Original query image
            detections: List of bboxes in COCO format [x, y, width, height]
            support_images: Support examples for the target class
            class_name: Target class name
            confidence_threshold: Minimum confidence to keep detection

        Returns:
            Tuple of:
            - Filtered detections: List of dicts with bbox, confidence, decision
            - Statistics: Dict with verification counts
        """
        if not detections:
            return [], {
                "total_detections": 0,
                "verified_valid": 0,
                "verified_invalid": 0,
                "verification_errors": 0,
            }

        filtered_detections = []
        stats = {
            "total_detections": len(detections),
            "verified_valid": 0,
            "verified_invalid": 0,
            "verification_errors": 0,
            "confidence_threshold": confidence_threshold,
        }

        for idx, bbox in enumerate(detections):
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            # Crop detection region
            try:
                cropped = query_image.crop((x, y, x2, y2))
            except Exception as e:
                logger.warning(f"Failed to crop detection {idx}: {e}")
                stats["verification_errors"] += 1
                continue

            # Verify cropped detection
            verification = self.verify_detection(cropped, support_images, class_name)

            if verification.decision == "ERROR":
                stats["verification_errors"] += 1
                continue

            is_valid = (
                verification.decision == "YES"
                and verification.confidence >= confidence_threshold
            )

            if is_valid:
                stats["verified_valid"] += 1
                filtered_detections.append(
                    {
                        "bbox": bbox,
                        "confidence": verification.confidence,
                        "decision": verification.decision,
                        "reasoning": verification.reasoning,
                    }
                )
            else:
                stats["verified_invalid"] += 1
                logger.debug(
                    f"Detection {idx} filtered out: {verification.decision} "
                    f"({verification.confidence:.2f} < {confidence_threshold})"
                )

        logger.info(
            f"Verification complete for {class_name}: "
            f"{stats['verified_valid']} valid, {stats['verified_invalid']} invalid, "
            f"{stats['verification_errors']} errors"
        )

        return filtered_detections, stats
