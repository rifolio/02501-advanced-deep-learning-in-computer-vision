from config import settings
from data.dataloaders import get_coco_dataloader, get_coco_few_shot_dataloader, get_coco_oracle_shot_dataloader
from models.qwen import Qwen2_5_VL
from models.internVL import InternVL
from models.grounding_dino import GroundingDINO
from models.vlm_dino_fusion import VLMDINOFusion

from pipeline import Experiment, FewShotExperiment

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _get_model():
    model_name = settings.model_name.lower()
    if model_name == "qwen":
        return Qwen2_5_VL(device=settings.device)
    if model_name == "internvl":
        return InternVL(device=settings.device, model_id=settings.internvl_model_id)
    if model_name == "grounding_dino":
        return GroundingDINO(device=settings.device)    
    if model_name == "vlm_dino_fusion":
        return VLMDINOFusion(device=settings.device)    
    raise ValueError("model_name must be one of: qwen, internvl, grounding_dino, vlm_dino_fusion")


def _validate_runtime_configuration() -> None:
    mode = settings.experiment_mode.lower()
    prompt_strategy = settings.prompt_strategy.lower()
    if mode == "few_shot" and prompt_strategy == "verification":
        raise ValueError(
            "Invalid configuration: prompt_strategy='verification' is not supported in the bbox "
            "detection few-shot pipeline. Use a detection strategy "
            "('side_by_side', 'cropped_exemplars', 'text_from_vision', 'set_of_mark', "
            "'vlm_text_generation') or run the dedicated verification pipeline."
        )
    if mode == "oracle_shot" and prompt_strategy != "oracle_shot":
        raise ValueError(
            f"experiment_mode='oracle_shot' requires prompt_strategy='oracle_shot', got '{prompt_strategy}'."
        )


def main():
    _validate_runtime_configuration()
    mode = settings.experiment_mode.lower()
    is_few_shot = mode == "few_shot"
    is_oracle = mode == "oracle_shot"

    if is_oracle:
        project_name = "VLM_OracleShot_Detection"
    elif is_few_shot:
        project_name = "VLM_FewShot_Detection"
    else:
        project_name = "VLM_ZeroShot_Detection"

    dataset = settings.data_dir.split("/")[-1]

    if is_oracle:
        test_loader = get_coco_oracle_shot_dataloader()
    elif is_few_shot:
        test_loader = get_coco_few_shot_dataloader(k_shot=settings.k_shot)
    else:
        test_loader = get_coco_dataloader()

    experiment_config = {
        "test_loader": test_loader,
        "model": _get_model(),
        "dataset": dataset,
    }

    experiment_cls = FewShotExperiment if (is_few_shot or is_oracle) else Experiment
    experiment = experiment_cls(project_name=project_name, config=experiment_config)
    experiment.run_evaluation()

if __name__ == '__main__':
    main()