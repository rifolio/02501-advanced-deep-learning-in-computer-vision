from config import settings
from data.dataloaders import coco_val_loader
from models.qwen import Qwen2_5_VL
from models.internVL import InternVL2_5_8B
from models.grounding_dino import GroundingDINO

from pipeline import Experiment

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

def main():
    project_name = 'VLM_ZeroShot_Detection'
    dataset = settings.data_dir.split('/')[-1]

    # Instantiate the model
    intern_VL_8B = InternVL2_5_8B(device=settings.device)
    dino_model = GroundingDINO(device=settings.device) # <-- Instantiate DINO
    qwen_model = Qwen2_5_VL(device=settings.device)
    # Configure the experiment
    experiment_config = {
        'test_loader': coco_val_loader,
        'model': qwen_model,
        #'model': dino_model,
        'dataset': dataset
    }

    experiment = Experiment(
        project_name=project_name,
        config=experiment_config
    )

    # Run the pipeline
    experiment.run_evaluation()

if __name__ == '__main__':
    main()