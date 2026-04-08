from datasets import FrameImageDataset, FrameVideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

root_dir = 'data/ufc10'

transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
frameimage_dataset = FrameImageDataset(
    root_dir = root_dir,
    split='val',
    transform=transform,
)

frameimage_loader = DataLoader(
    frameimage_dataset,
    batch_size=8,
    shuffle=False
    )

for images, labels in frameimage_loader:
    print(images.shape, labels.shape) # [batch, channels, height, width]

    print(images[0], labels[0])
    break