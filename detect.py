from models import Model
from tool import *
import argparse
import torchvision
from torch.utils import data
import visdom
import scipy
from dataset import Dataset, Datasetdis

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='rice', type=str, help='dataset')
parser.add_argument('--bs', default=3, type=int, help='batch size')
parser.add_argument('--data_path', default=r'.\\URC', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'./checkpoint/RiceNet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default='0', help='assign device')
args = parser.parse_args()
vis = visdom.Visdom()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu.strip()


model_density = Model().cuda()
model_density.load_state_dict(torch.load('checkpoint\\RiceNet\\checkpoint_best.pth'))

model_size = Model(gap = True).cuda()
model_size.load_state_dict(torch.load('checkpoint\\size_pth\\checkpoint_bestdis.pth'))
model_density.eval()
model_size.eval()


checkpoint = torch.load('checkpoint\\size_pth\\checkpoint_latestdis.pth')
vis.line(win='distraon_loss',X=checkpoint['epoch_list'], Y=checkpoint['train_maelist'], opts=dict(title='distrainloss'))
vis.line(win='mae',X=checkpoint['epoch_list'], Y=checkpoint['test_maelist'], opts=dict(title='dismae'))


test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
crop = torchvision.transforms.RandomCrop((400,400))

for i, (images, gt,dis, img_path) in enumerate(test_loader):
    torch.cuda.empty_cache()
    fname = os.path.basename(img_path[0])
    images = images.cuda()
    
    pre_dis = model_size(images)
    label = h5py.File(img_path[0].replace('jpg','h5').replace('imgs_4','tep_density'), 'r')
    predict = torch.from_numpy(np.array(label['density'], dtype=np.float32))
    count, kpoint = Center_counting(images, predict, dis, img_path)
    
    
    ori_Img_data, Img_data = generage_boxx(img_path, kpoint, pre_dis)
    cv2.imwrite('.\\plant_size_output\\' + fname, Img_data)
   