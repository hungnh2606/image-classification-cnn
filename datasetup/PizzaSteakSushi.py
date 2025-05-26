import os.path

from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose

class PizzaSteakSushiDataset(Dataset):
    def __init__(self, root, train = True, transform=None):
        if train:
            mode = "train"
        else:
            mode = "test"

        self.root = os.path.join(root, mode)
        print(os.listdir(self.root))
        self.categories = ['pizza', 'steak', 'sushi']
        self.transform = transform

        self.image_path = []
        self.labels = []

        for i, category in enumerate(self.categories):
            data_files_path = os.path.join(self.root, category)
            print(data_files_path)
            for file_name in os.listdir(data_files_path):
                file_path = os.path.join(data_files_path, file_name)
                self.image_path.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        # if self.transform:
        #     image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    root = "D:\Image classification\data\Food\pizza_steak_sushi"
    dataset = PizzaSteakSushiDataset(root, train=True)
    print(dataset.root)
    image, labels = dataset.__getitem__(100)
    print(image.show())
    print(labels)
    # transform = Compose([
    #     Resize((224, 224)),
    #     ToTensor(),
    # ])
    #
    # training_data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=16,
    #     num_workers=4,
    #     shuffle=True,
    #     drop_last=False
    # )
    #
    # for images, labels in training_data_loader:
    #     print(images.shape)
    #     print(labels)

