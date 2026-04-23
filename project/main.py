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

# Optional but recommended for V100s when using fp16
# torch.backends.cuda.matmul.allow_tf32 = False 
# torch.backends.cudnn.allow_tf32 = False
# project_name = 'VLM_ZeroShot_Detection'
# dataset = settings.data_dir.split('/')[-1]

# # Instantiate the model
# #qwen_model = Qwen2_5_VL(device=settings.device)
# intern_VL_1B = InternVL2_5_1B(device=settings.device)

# # Configure the experiment
# # experiment_config = {
# #     'test_loader': coco_val_loader,
# #     'model': qwen_model,
# #     'dataset': dataset
# # }

# experiment_config = {
#     'test_loader': coco_val_loader,
#     'model': intern_VL_1B,
#     'dataset': dataset
# }

# experiment = Experiment(
#     project_name=project_name,
#     config=experiment_config
# )

# # Run the pipeline
# experiment.run_evaluation()

def _get_model():
    model_name = settings.model_name.lower()
    if model_name == "qwen":
        return Qwen2_5_VL(device=settings.device)
    if model_name == "internvl":
        return InternVL2_5_8B(device=settings.device)
    if model_name == "grounding_dino":
        return GroundingDINO(device=settings.device)    if model_name == "vlm_dino_fusion":
        return VLMDINOFusion(device=settings.device)    raise ValueError("model_name must be one of: qwen, internvl, grounding_dino")


def main():
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