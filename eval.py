import torch
from torch.utils import data
from dataset import Dataset
from models import Model
import os
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='rice', type=str, help='dataset')
parser.add_argument('--data_path', default=r'.\\URC', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'.\\checkpoint\\RiceNet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)
checkpoint1 = {}
checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
checkpoint1['model'] = checkpoint
model.load_state_dict(checkpoint1['model'])

gt_count = []
pre_count = []
model.eval()

with torch.no_grad():
    mae, mse = 0.0, 0.0
    mae1, mse1 = 0.0, 0.0
    tep = {}
    for i, (images, gt, img_path,_) in enumerate(test_loader):
        images = images.to(device)
        gt = gt.to(device)
        predict, _ = model(images)
        gt_count.append(gt.item())
        pre_count.append(predict.sum().item())
        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        mae += torch.abs(predict.sum() - gt).item()
        mse += ((predict.sum() - gt) ** 2).item()
        mae1 += torch.abs((predict.sum() - gt)/gt).item()
        mse1 += (((predict.sum() - gt)/gt) ** 2).item()
   
    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    rmae = mae1 / len(test_loader)
    rmse = (mse1 / len(test_loader))** 0.5
    print('MAE:', mae, 'MSE:', mse,"rmae:",rmae,'rmse:',rmse)
    torch.save({
            'gt': np.array(gt_count),
            'pre': np.array(pre_count),
            'name': 'rice'
        }, './n_2.pth')