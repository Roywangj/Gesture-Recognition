import torch
from skimage import transform as tranf
import cv2
import os

class GestureData(torch.utils.data.Dataset):
    def __init__(self, root, train, transform, target_transform=None):
        self.train = train
        if self.train == True:
            f = open(root + "/train.txt", "r")
            print("READ TRAIN LIST FILE SUCCESS.")
        else:
            f = open(root + "/test.txt", "r")
            print("READ TEST LIST FILE SUCCESS.")

        self.records = f.readlines()
        self.transform = transform

    def __getitem__(self, index):
        record = self.records[index]
        img_path = record.split(",")[0]
        label = int(record.split(",")[1].strip())
        image = cv2.imread(img_path)
        if not image.shape == (100, 100, 3):
            # print(image.shape, img_path)
            image = tranf.resize(image, (100, 100))
            # print('CHANGED')
        if self.transform:
            image = self.transform(image)
        else:
            image = image.transpose(2, 1, 0)
            image = torch.tensor(image, dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.records)

