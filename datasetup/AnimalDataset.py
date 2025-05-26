import os.path

import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose

class AnimalDataset(Dataset):

    def __init__(self, root, train=True, transform=None):
        if train:
            mode = "train"
        else:
            mode = "test"
        self.root = os.path.join(root, mode)
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.transform = transform

        self.image_path =[]
        self.labels = []

        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)
            print(data_file_path)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path,file_name)
                # print(file_path)
                self.image_path.append(file_path)
                self.labels.append(i)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        # open with PIL
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # open with open cv
        # image = cv2.imread(image_path)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    root = "D:\Image classification\data\Animal"
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = AnimalDataset(root, train=True, transform=transform)
    training_data_loader = DataLoader(
        dataset = dataset,
        batch_size= 16,
        num_workers= 4,
        shuffle= True,
        drop_last=False
    )
    for images, labels in training_data_loader:
        print(images.shape)
        print(labels)
    # image, label = dataset.__getitem__(548)
    # print(image.show())
    # # print(cv2.imshow("image", image))
    # # cv2.waitKey(0)
    # print(label)