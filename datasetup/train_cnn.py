import os.path
import shutil

import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from torchvision import models
from FoodVietNam import FoodVietNamDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default='D:\Image classification\data\FoodVietNam', help="Path data")
    parser.add_argument("--epochs","-e",type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b" ,type=int, default=16, help="Number of batch-size")
    parser.add_argument("--image-size","-i" ,type=int, default=224, help="Size of image")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t",type=str, default="trained_model")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transforms = Compose([
        RandomAffine(
            degrees=(-5,5),
            translate=(0.15,0.15),
            scale=(0.85,1.15),
            shear=5
        ),
        Resize((args.image_size,args.image_size)),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.25,
            hue=0.05
        ),
        ToTensor()
    ])
    test_transforms = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])
    # train_dataset = AnimalDataset(root=args.root,train=True, transform=train_transforms)
    train_dataset = FoodVietNamDataset(root=args.root,train=True, transform=train_transforms)

    # image,label = train_dataset.__getitem__(1234)
    # image = (torch.permute(image, (1,2,0))*255.).numpy().astype(np.uint8)
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # cv2.imshow("test image", image)
    # cv2.waitKey(0)
    # exit(0)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = FoodVietNamDataset(root=args.root,train=False, transform=test_transforms)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)
    model = models.efficientnet_b0(pretrained=True)
    for name, param in model.named_parameters():
        if "classifier" not in name and "features.6" not in name and "features.7" not in name:
            param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, 36)
    )
    # model = SimpleCNN(num_classes = 10).to(device)
    # use transfer learning
    # for name, param in model.named_parameters():
    #     if "fc." not in name and "layer4." not in name:
    #         param.requires_grad = False
    #     print(name, param.requires_grad)
    model = model.to(device)
    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    num_iters = len(train_dataloader)

    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images) # forward
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs,iter+1,num_iters,loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch * num_iters + iter)

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions),class_names=test_dataset.categories,epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, accuracy))
        writer.add_scalar("Train/Accuracy", accuracy, epoch)
        #torch.save(model.state_dict(),"{}/last_cnn.pt".format(args.trained_models))
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint,"{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
#        print(classification_report(all_labels, all_predictions))


