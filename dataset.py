from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional
import scipy.io as sio   
import random
class Dataset(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset

        if dataset == 'rice':
            if is_train:
                dataset_img = os.path.join('train', 'imgs_4')
                dataset_label = os.path.join('train', 'new_data_4')
            else:
                dataset_img = os.path.join('test', 'imgs_4')
                dataset_label = os.path.join('test', 'new_data_4')

        self.image_list = glob.glob(os.path.join(data_path, dataset_img, '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, dataset_label, '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        labeldis = h5py.File(self.label_list[index].replace('new_data_4','dis_data_4'), 'r')
        density = np.array(label['density'], dtype=np.float32)
        attention = np.array(label['attention'], dtype=np.float32)
        gt = np.array(label['kpoint'], dtype=np.float32).sum().reshape((1,1))
        dis = np.array(labeldis['dis'], dtype=np.float32)
        trans = Transforms((0.8, 1.2), (320, 320), 2, (0.5, 1.5), self.dataset)   
        if self.is_train:
            image, density, attention, kpoint, s_and_e = trans(image, density, attention, self.image_list[index])
            return image, density, attention, kpoint, s_and_e
        else:
            height, width = image.size[1], image.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt, dis, self.image_list[index]

    def __len__(self):
        return len(self.image_list)

class Datasetdis(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset
        if is_train:
            dataset_img = os.path.join('train', 'imgs_4')
            dataset_label = os.path.join('train', 'dis_data_4')
        else:
            dataset_img = os.path.join('test', 'imgs_4')
            dataset_label = os.path.join('test', 'dis_data_4')
        self.image_list = glob.glob(os.path.join(data_path, dataset_img, '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, dataset_label, '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        dis = np.array(label['dis'], dtype=np.float32)

        height, width = image.size[1], image.size[0]
        h, w = 400, 400
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
       
        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       
        return image, dis

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    train_dataset = Dataset(r'D:\dataset', 'SHA', True)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for image, label, att in train_loader:
        print(image.size())
        print(label.size())
        print(att.size())

        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(label.squeeze(), cmap='jet')
        plt.subplot(1, 3, 3)
        plt.imshow(att.squeeze(), cmap='jet')
        plt.show()
