from config import settings
from data.dataloaders import get_coco_dataloader, get_coco_few_shot_dataloader
from models.qwen import Qwen2_5_VL
from models.internVL import InternVL2_5_8B
from models.grounding_dino import GroundingDINO
from models.vlm_dino_fusion import VLMDINOFusion

from pipeline import Experiment, FewShotExperiment

import torch
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _get_model():
    model_name = settings.model_name.lower()
    if model_name == "qwen":
        return Qwen2_5_VL(device=settings.device)
    if model_name == "internvl":
        return InternVL2_5_8B(device=settings.device)
    if model_name == "grounding_dino":
        return GroundingDINO(device=settings.device)    
    if model_name == "vlm_dino_fusion":
        return VLMDINOFusion(device=settings.device)    
    raise ValueError("model_name must be one of: qwen, internvl, grounding_dino")


def _validate_runtime_configuration() -> None:
    is_few_shot = settings.experiment_mode.lower() == "few_shot"
    prompt_strategy = settings.prompt_strategy.lower()
    if is_few_shot and prompt_strategy == "verification":
        raise ValueError(
            "Invalid configuration: prompt_strategy='verification' is not supported in the bbox "
            "detection few-shot pipeline. Use a detection strategy "
            "('side_by_side', 'cropped_exemplars', 'text_from_vision', 'set_of_mark', "
            "'vlm_text_generation') or run the dedicated verification pipeline."
        )


def main():
    _validate_runtime_configuration()
    is_few_shot = settings.experiment_mode.lower() == "few_shot"
    project_name = "VLM_FewShot_Detection" if is_few_shot else "VLM_ZeroShot_Detection"
    dataset = settings.data_dir.split("/")[-1]

    experiment_config = {
        "test_loader": (
            get_coco_few_shot_dataloader(k_shot=settings.k_shot)
            if is_few_shot
            else get_coco_dataloader()
        ),
        "model": _get_model(),
        "dataset": dataset,
    }

    experiment_cls = FewShotExperiment if is_few_shot else Experiment
    experiment = experiment_cls(project_name=project_name, config=experiment_config)
    experiment.run_evaluation()

if __name__ == '__main__':
    main()