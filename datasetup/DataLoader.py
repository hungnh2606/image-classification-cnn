from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Resize, Compose

from FoodVietNam import FoodVietNamDataset

if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    training_data = FoodVietNamDataset(root="D:\Image classification\data\FoodVietNam", train=True, transform=transform)

    training_data_loader = DataLoader(
        dataset=training_data,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        drop_last=False
    )

    for images, labels in training_data_loader:
        print(images.shape)
        print(labels)