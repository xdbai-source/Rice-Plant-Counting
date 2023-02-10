from sklearn import model_selection
import torch
from torch import nn
from torch import optim
from torch.utils import data
from dataset import Dataset, Datasetdis    
from models import Model                   
import os
from datetime import datetime
import argparse
from torchvision import transforms
import visdom
import matplotlib.pyplot as plt
from torchvision.ops import RoIPool, roi_pool
from torch.optim.lr_scheduler import StepLR
import log_utils as log_utils
import random
from tool import *
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    
    torch.backends.cudnn.deterministic = True
seed_torch(42)


def adjust_learning_rate(optimizer, epoch):
    if epoch>150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    if epoch>250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    if epoch>260 and epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            if param_group['lr'] == 1e-4:
                param_group['lr'] = 1e-5
            else:
                param_group['lr'] = 1e-4


parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=3, type=int, help='batch size')
parser.add_argument('--epoch', default=500, type=int, help='train epochs')
parser.add_argument('--dataset', default='rice', type=str, help='dataset')
parser.add_argument('--data_path', default=r'I:\\dataset\\dataset_rice\\rice-jiangxi\\datass', type=str, help='path to dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--load', default=False, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default=r'./checkpoint/RiceNet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default='2', help='assign device')
transform1 = transforms.Normalize(mean=[-(0.485/0.229), -(0.456/0.224), -(0.406/0.225)],       
                                        std=[1/0.229, 1/0.224, 1/0.225])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu.strip()        


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
logger = log_utils.get_logger(os.path.join(args.save_path, 'train-{:s}.log'.format(time_str)))
log_utils.print_config(vars(args),logger)    


train_dataset = Dataset(args.data_path, args.dataset, True)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    logger.info('using {} gpus'.format(device_count))
else:
    raise Exception("gpu is not available")

model = Model().to(device)  

mseloss = nn.MSELoss(reduction='sum').to(device)
bceloss = nn.BCELoss(reduction='sum').to(device)          


optimizer = optim.Adam(model.parameters(), lr=args.lr)    

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  
train_loss_list = []
epoch_list = []
test_error_list = []
vis = visdom.Visdom()


if args.load:
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_mae = 999999
    start_epoch = 0


for epoch in range(start_epoch, start_epoch + args.epoch):
    
    scheduler.step()
    loss_avg, loss_att_avg, loss_fine_avg, loss_pre_avg = 0.0, 0.0, 0.0, 0.0
    model.train()
    for i, (images, density, att, kpoint, s_and_e) in enumerate(train_loader):
        images = images.to(device)
        density = density.to(device)
        att = att.to(device)
        kpoint = kpoint.to(device)
        s_and_e = s_and_e.to(device)
        outputs, attention = model(images)
       
        loss = mseloss(outputs, density) / args.bs   
        
        loss_att = bceloss(attention, att) * 0.1     

        
        pre_loss = torch.abs((outputs.view(outputs.shape[0],outputs.shape[1],-1)*s_and_e).view(outputs.shape[0],-1).sum(1)-density.view(density.shape[0],-1).sum(1)).mean()* 0.1

        loss_sum = loss + loss_att + pre_loss          

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
   
        loss_avg += loss.item()
        loss_att_avg += loss_att.item()
        loss_pre_avg += pre_loss.item()
    logger.info("Epoch:{}, Loss:{:.5f}, Attloss: {:.5f},Preloss: {:.5f}".format(epoch, loss_avg/len(train_loader), loss_att_avg/len(train_loader), loss_pre_avg/len(train_loader)))
    train_loss_list.append(loss_avg/len(train_loader))
    epoch_list.append(epoch)
    vis.line(win='rice2_2_train',X=epoch_list, Y=train_loss_list, opts=dict(title='rice2_2_trainloss'))
    
    
    model.eval()
    with torch.no_grad():
        mae, mse = 0.0, 0.0
        for i, (images, gt,_,_) in enumerate(test_loader):
            images = images.to(device)
            gt = gt.to(device)
            predict, _ = model(images)
            mae += torch.abs(predict.sum() - gt).item()
            mse += ((predict.sum() - gt) ** 2).item()
        mae /= len(test_loader)
        mse /= len(test_loader)
        mse = mse ** 0.5
        test_error_list.append(mae)
        vis.line(win='rice2_2test_loss',X=epoch_list, Y=test_error_list, opts=dict(title='rice2_2testloss'))
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(args.save_path, 'checkpoint_best.pth'))

        logger.info('Epoch:{}, MAE:{:.2f}, MSE:{:.2f}, best mae:{:.2f}'.format(epoch, mae, mse, best_mae))
        save_path = os.path.join(args.save_path, 'checkpoint_latest.pth')
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'train_maelist': train_loss_list,
            'test_maelist': test_error_list,
            'epoch_list' : epoch_list,
            'best mae' : best_mae
        }, save_path)


model = Model(gap=True).to(device)                     
model.load_state_dict(torch.load(os.path.join('checkpoint\\rice_8.5', 'checkpoint_best.pth')),strict = False) 
optimizer = optim.Adam(model.parameters(), lr=args.lr) 


train_dataset = Datasetdis(args.data_path, args.dataset, True)
traindis_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Datasetdis(args.data_path, args.dataset, False)
testdis_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
startdis_epoch = 0
allepoch = 500
best_dis = 99999999999
traindis_loss_list = []
epochdis_list = []
testdis_list = []

for epoch in range(startdis_epoch, allepoch):
    loss_dis_avg = 0.0
    model.train()
    for i, (images,dis) in enumerate(traindis_loader):
        images = images.to(device)
        dis = dis.to(device)
        pro_dis = model(images)

        loss_scale = torch.mean(abs(pro_dis - dis))

        optimizer.zero_grad()
        loss_scale.backward()
        optimizer.step()

        loss_dis_avg += loss_scale.item()
    logger.info("DisEpoch:{}, Loss:{:.2f}".format(epoch, loss_dis_avg/len(traindis_loader)))
    traindis_loss_list.append(loss_dis_avg/len(traindis_loader))
    epochdis_list.append(epoch)
    vis.line(win='dis_train',X=epochdis_list, Y=traindis_loss_list, opts=dict(title='dis_loss'))
    
    
    model.eval()
    with torch.no_grad():
        mae, mse = 0.0, 0.0
        for i, (images, dis) in enumerate(testdis_loader):
            images = images.to(device)
            dis = dis.to(device)
            pre_dis = model(images)
            mae += torch.abs(pre_dis - dis).item()
            mse += ((pre_dis - dis) ** 2).item()
        mae /= len(test_loader)
        mse /= len(test_loader)
        mse = mse ** 0.5
        testdis_list.append(mae)
        vis.line(win='dismae_loss',X=epochdis_list, Y=testdis_list, opts=dict(title='dismaeloss'))
        if mae < best_dis:
            best_dis = mae
            torch.save(model.state_dict(), os.path.join(args.save_path, 'checkpoint_bestdis.pth'))
        logger.info('disEpoch:{}, disMAE:{:.2f}, disMSE:{:.2f}, best dis mae:{:.2f}'.format(epoch, mae, mse, best_dis))
        save_path = os.path.join(args.save_path, 'checkpoint_latestdis.pth')
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'train_maelist': traindis_loss_list,
            'test_maelist': testdis_list,
            'epoch_list' : epochdis_list,
            'best mae' : best_dis
        }, save_path)






