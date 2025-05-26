import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose

class FoodVietNamDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            mode = "train"
        else:
            mode = "test"
        self.root = os.path.join(root, mode)
        # print(os.listdir(self.root))
        self.categories = ['Banh beo', 'Banh bo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cong', 'Banh cuon', 'Banh da lon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tai heo', 'Banh tet', 'Banh tieu', 'Banh trang nuong', 'Banh trung thu', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to', 'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo']
        # print(len(self.categories))
        self.transform = transform

        self.image_path = []
        self.labels = []

        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)
            # print(data_file_path)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.image_path.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        images_path = self.image_path[item]
        images = Image.open(images_path).convert('RGB')
        if self.transform:
            images = self.transform(images)
        label = self.labels[item]
        return images, label

if __name__ == '__main__':
    root = "D:\Image classification\data\FoodVietNam"
    dataset = FoodVietNamDataset(root, train=False)
    print(dataset.root)
    print(len(dataset))
    image, labels = dataset.__getitem__(100)
    print(image.show())
    print(labels)


    # 22531