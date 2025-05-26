from argparse import ArgumentParser

import cv2
import numpy as np
from torch import nn
from torchvision import models
from FoodVietNam import FoodVietNamDataset
import torch


def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image-path", "-p", type=str, default="D:/Image classification/test/bun dau 4.jpg", help="Path image test")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--checkpoint", "-c", type=str, default="D:/Image classification/datasetup/trained_model/best_cnn.pt")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    # categories = ['cat', 'dog']
    categories =  ['Banh beo', 'Banh bo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cong', 'Banh cuon', 'Banh da lon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tai heo', 'Banh tet', 'Banh tieu', 'Banh trang nuong', 'Banh trung thu', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to', 'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo']
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model = SimpleCNN(num_classes=10).to(device)
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, 36)
    )
    model = model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint found")
        exit(0)

    model.eval()
    ori_image = cv2.imread(args.image_path)
    print(ori_image)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2,0,1))/255.0
    image = image[None, :,:,:] # 1 * 3 * 224 * 224
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax(dim=1)


    with torch.no_grad():
        output = model(image)
        probs = softmax(output)

    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    print("The test image is about {} with confident score of {}".format(predicted_class,probs[0,max_idx]))
    cv2.imshow("{}:{: .2f}%".format(predicted_class,probs[0,max_idx]*100),ori_image)
    cv2.waitKey(0)



