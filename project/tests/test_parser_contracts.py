import unittest

from models.internVL import InternVL
from models.qwen import Qwen2_5_VL


class ParserContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Bypass heavyweight model initialization; parser methods do not depend on weights.
        self.internvl = InternVL.__new__(InternVL)
        self.qwen = Qwen2_5_VL.__new__(Qwen2_5_VL)

    def test_internvl_parses_historical_integer_box_format(self):
        text = "<ref>person</ref><box>[[100, 200, 500, 700]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=1000, img_height=1000)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [100.0, 200.0, 400.0, 500.0])

    def test_internvl_fallback_accepts_decimal_boxes(self):
        text = "<box>[[100.5, 200.25, 500.75, 700.5]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=1000, img_height=1000)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 100.5)
        self.assertAlmostEqual(boxes[0][1], 200.25)

    def test_internvl_rejects_unrelated_numeric_lists(self):
        text = "Top-4 confidence values: [0.98, 0.76, 0.52, 0.33]. Also IDs: [1, 2, 3, 4]."
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=640, img_height=480)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(boxes, [])

    def test_internvl_clamps_boxes_to_image_bounds(self):
        text = "<box>[[-100, -50, 1200, 1400]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=100, img_height=200)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [0.0, 0.0, 100.0, 200.0])

    def test_internvl_discards_degenerate_boxes_after_clamp(self):
        text = "<box>[[500, 500, 400, 600], [1200, 100, 1300, 200], [10, 20, 110, 220]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=1000, img_height=1000)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [10.0, 20.0, 100.0, 200.0])

    def test_qwen_parses_current_corner_format(self):
        text = "<|box_start|>(28, 56), (280, 420)<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=280, img_height=420)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [28.0, 56.0, 252.0, 364.0])

    def test_qwen_fallback_accepts_decimal_corner_format(self):
        text = "<|box_start|>([28.5, 56.25]), ([280.5, 420.75])<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=280, img_height=420)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 28.5)
        self.assertAlmostEqual(boxes[0][1], 56.25)

    def test_qwen_list_format_applies_normalized_scaling(self):
        """List-format [[x1,y1,x2,y2]] uses [0,1000] normalized coords, not pixel space."""
        text = "[[250, 250, 750, 750]]"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=640, img_height=480)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 160.0)   # 250/1000 * 640
        self.assertAlmostEqual(boxes[0][1], 120.0)   # 250/1000 * 480
        self.assertAlmostEqual(boxes[0][2], 320.0)   # (750-250)/1000 * 640
        self.assertAlmostEqual(boxes[0][3], 240.0)   # (750-250)/1000 * 480

    def test_qwen_native_format_still_uses_ratio_correction(self):
        """Native <|box_start|> format applies pixel-space ratio correction, not /1000."""
        text = "<|box_start|>(28, 56), (280, 420)<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=280, img_height=420)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [28.0, 56.0, 252.0, 364.0])

    def test_qwen_list_format_drops_degenerate_boxes(self):
        """Degenerate boxes (xmax <= xmin after normalization) should be dropped."""
        text = "[[500, 500, 500, 800], [100, 100, 400, 400]]"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=640, img_height=480)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 64.0)    # 100/1000 * 640
        self.assertAlmostEqual(boxes[0][1], 48.0)    # 100/1000 * 480

    def test_qwen_native_format_drops_degenerate_boxes(self):
        """Native-format degenerate boxes (xmax <= xmin) should be dropped."""
        text = "<|box_start|>(200, 100), (100, 300)<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=640, img_height=480)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(boxes, [])


if __name__ == "__main__":
    unittest.main()
