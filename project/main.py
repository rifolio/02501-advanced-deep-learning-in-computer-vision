from config import settings
from data.dataloaders import coco_val_loader
from models.qwen import Qwen2_5_VL
from pipeline import Experiment

project_name = 'VLM_ZeroShot_Detection'
dataset = settings.data_dir.split('/')[-1]

# Instantiate the model
qwen_model = Qwen2_5_VL(device=settings.device)

# Configure the experiment
experiment_config = {
    'test_loader': coco_val_loader,
    'model': qwen_model,
    'dataset': dataset
}

experiment = Experiment(
    project_name=project_name,
    config=experiment_config
)

# Run the pipeline
experiment.run_evaluation()