from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import random
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import torch.nn as nn
class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma
        self.dataset = dataset
        
    def __call__(self, image, density, attention, img_path):   
        
        height, width = image.size[1], image.size[0]

        scale = random.uniform(self.scale[0], self.scale[1])
        height = round(height * scale)
        width = round(width * scale)
        image = image.resize((width, height), Image.BILINEAR)
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)
        
        
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]
        attention = attention[dh:dh + h, dw:dw + w]
       
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1]
            attention = attention[:, ::-1]
        
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)

        

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
                               interpolation=cv2.INTER_LINEAR)
        keep = nn.functional.max_pool2d(functional.to_tensor(density), (17,17), stride=1, padding=8).numpy()
        keep = (keep == density).astype(np.float32) 
        kpoint = keep * density

        atte = attention > 0.001
        attention = atte.astype(np.float32)
        kpp = kpoint > 0.001
        kpoint = kpp.astype(np.float32)    

        subdensity = np.maximum(density.reshape(1, density.shape[0]*density.shape[1]), 0)
        
        empty = subdensity <= 0          
        solid  = subdensity > 0         
        empty = solid.astype(np.float32)*0.2+empty.astype(np.float32)
        s_and_e = np.concatenate((solid.astype(np.float32)*0.8,empty),axis=0)
        

        s_and_e = s_and_e.copy()                
        density = np.reshape(density, [1, density.shape[0], density.shape[1]])
        attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])
        kpoint = kpoint.copy()
        return image, density, attention, kpoint, s_and_e
