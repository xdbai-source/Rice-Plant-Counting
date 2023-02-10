import torch 
import os 
import copy
import matplotlib.pyplot as plt
from torch import float32, nn, Tensor
import numpy as np
import cv2
import math
import h5py
from torchvision import transforms
import scipy
import scipy.spatial

def Center_counting(images,input,dis, img_path):
    
    h,w = input.shape[2],input.shape[3]
    transform = transforms.Compose([transforms.Normalize(mean=[-(0.485/0.229), -(0.456/0.224), -(0.406/0.225)],
                                        std=[1/0.229, 1/0.224, 1/0.225])])
    
  
    max_list = input.view(input.size(0), -1).max(1)[0]
        
    dis = int((dis/2*0.3)//2*2+1)
    keep = nn.functional.max_pool2d(input, (dis,dis), stride=1, padding=int((dis-1)/2))
      
    keep = (keep == input).float()
    input = keep * input
    
    fiter = (75.0 / 255.0 * max_list)
  
   
    input[input < fiter] = 0
    input[input > 0] = 1
    
    
    if max_list < 0.001:
        input = input * 0
    count = input.sum()
    return count, input

    
def generage_boxx(img_path, kpoint, sigma):
    Img_data = cv2.imread(img_path[0])
    ori_Img_data = Img_data.copy()
    height, width = Img_data.shape[0], Img_data.shape[1]
    height = round(height / 16) * 16
    width = round(width / 16) * 16
    Img_data = cv2.resize(Img_data, (width, height))
    coor = torch.nonzero(kpoint[0][0]).cpu().data.numpy()
    mat = coor.copy()
    k = min(len(mat),3)
    tree = scipy.spatial.KDTree(mat.copy(), leafsize=1024)    
    distances,_ = tree.query(mat, k=len(mat))
    distances_mean = 0.8*np.mean(distances[:,1:k],axis=1)*2
    if sigma!= 0:
        for j in range(len(distances_mean)):
            y = int(coor[j][0]*2)
            x = int(coor[j][1]*2)
            if sigma[0][0]<(distances_mean[j]):
                dis = sigma[0][0]  
            else:
                dis = torch.from_numpy(np.array(distances_mean[j]))
            Img_data = cv2.rectangle(Img_data, ((x - dis/2),(y-dis/2)),((x + dis/2),(y + dis/2)), (255, 255, 255), 2)
            Img_data = cv2.circle(Img_data, (x,y), 2, (0,0,255), 2)
    else:
        Img_data = ori_Img_data.copy()
    return ori_Img_data, Img_data

def Img_show(args,vis,img_path,es_map):
    img = cv2.imread(img_path[0])
    gt_path = img_path[0].replace('.jpg','.h5').replace('images','new_data')
    gt_file = h5py.File(gt_path,'r')
    gt_dmap = np.asarray(gt_file['density'])
    es_map = es_map[0][0].cpu().data
    vis.image(win=3,img=img.transpose((2,0,1)),opts=dict(title='img'))
    vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_map')) 
    vis.heatmap(win=5,X=np.flipud(gt_dmap/(gt_dmap.max())*255),opts=dict(title='gt_heatmap'))
    vis.image(win=6,img=es_map/(es_map.max())*255,opts=dict(title='es_map')) 
    vis.heatmap(win=7,X=np.flipud(es_map/(es_map.max())*255),opts=dict(title='es_heatmap'))