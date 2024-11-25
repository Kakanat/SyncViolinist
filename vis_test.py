import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from pathlib import Path
import yaml
import wandb
import glob
import joblib
import random
import os 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pathlib 
import sys
import wandb
from metrics import l1l1_loss,l1l2_loss,l2l1_loss,l2l2_loss

from model import Model
import metrics
from data import KeypsPPDataset, Normalize
from utils import sort_seq, save_video

sys.path.append('../')
from motionsynth_code.motion import BVH as BVH 
from motionsynth_code.motion.Animation import positions_global
from QuarterNet import skeleton
from QuarterNet.skeleton import Skeleton as Skeleton
from QuarterNet import visualization as vis 
from QuarterNet import visualization_vis4 as vis4 

COLORFULL = True
body_part = 'left_hand'
METRICS = {'L1':[], 'L1RH':[], 'L1LH':[], 'L1LF':[], 'DTW':[], 'DTWRH':[], 'DTWLH':[], 'DTWLF':[],
        'BowX':[], 'BowY':[], 'BowZ':[], 'BowAve':[]}

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def joints_num(parts:str):
    if parts == 'left_hand':
        return [value for value in range(33,52)]#32かも？
    elif parts == 'right_hand':
        return [value for value in range(9,30)]#8かも？
    elif parts == 'both_hands':
        return [value for value in list(range(9, 30)) + list(range(33, 52))]
    elif parts == 'full_body':
        return [value for value in range(0,62)]
    elif parts == 'woboth_hands':
        return [value for value in list(range(0,10))+list(range(30,34))+list(range(52,62))]
    elif parts == 'left_hand2':
        return [value for value in range(32,52)]#肘込み
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
def transform_keyps(y, mean, std):
    y = std * y + mean
    return y

def test(net, device, test_loader,loss_fn,filename,joints):
    for i, data in enumerate(test_loader, 0):
            gt = data[2]
            gts.append(gt)
    gts = torch.squeeze(torch.cat(gts).reshape(len(gts), *gts[0].shape))
    gts = torch.reshape(gts, (-1, 62, 3))
    return gts

def create_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('parts', type = str, default = 'full_body')
    parser.add_argument('--pp', type = str, default = '../../data/pp_one_hot/')
    parser.add_argument('--aud', type = str, default = '../../data/aud/')
    parser.add_argument('--keyps', type = str, default = '../../data/keyps_aligned/')
    parser.add_argument('--visualize', action = 'store_true')
    parser.add_argument('--sklsource', type=str, default = '../../data/joint_aligned/konishi_001.jb')
    args = parser.parse_args()
    return args


def main():
    seed_everything()
    args = create_options()
    color = {}
    config = {}
    wav_dir = '/home/mic/Desktop/hiroki/shyakyou/MotionGeneration/data/wav_normalized/'
    aud_data = sorted(glob.glob(args.aud + '/*.jb'))
    joint_data = sorted(glob.glob(args.keyps + '/*.jb'))
    pp_data = sorted(glob.glob(args.pp + '/*.jb'))
    pp_all = ('string', 'fing', 'pos', 'ud')
    pp_all_dim = (4, 5, 12, 2)
    pp = [pp_all[i] for i in range(4) if int('1111'[i])==1] 
    pp_dim = sum([pp_all_dim[i] for i in range(4) if int('1111'[i])==1])
    for aud_d, joint_d, pp_d in zip(aud_data, joint_data, pp_data):   
        # load dataset
        transform = Normalize()
        test_data = KeypsPPDataset([aud_d], [joint_d], [pp_d], pp=pp, transform = transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
        pred_keyps = {}
        gts = []
        
        for i, data in enumerate(test_loader, 0):
            gt = data[2]
            gts.append(gt)
        gts = torch.squeeze(torch.cat(gts).reshape(len(gts), *gts[0].shape))
        gts = torch.reshape(gts, (-1, 62, 3))
        gts = gts[:,joints_num(args.parts),:]
        gts = gts.to('cpu').detach().numpy().copy()   
        # import IPython;IPython.embed();exit()
        gt_keyps = transform_keyps(gts, mean=gts.mean(), std=gts.std())
        if args.visualize:
            audiofile = wav_dir + pathlib.Path(aud_d).stem + '.wav'
            videofile = './demo/' + pathlib.Path(joint_d).stem + '.mp4'
            sklsource = joblib.load(args.sklsource)
            skl = Skeleton(sklsource[0].astype(np.float32), sklsource[1])
            pred_keyps[f'{len(config)}'] = gt_keyps
            color[f'{len(config)}'] = (100,100,0)#emerald
            joints = joints_num(body_part)
            # import IPython;IPython.embed();exit()
            gt_keyps = gt_keyps[:,joints,:]
            if COLORFULL:
                vis.color_joint_cv2(gt_keyps, skl,joints, 30.0, './demo/' + pathlib.Path(joint_d).stem + '_colorfull.mp4', audiofile,parts = body_part)
            else:
                vis4.render_joint_cv2(pred_keyps, skl,joints_num(args.parts), 30.0, videofile, audiofile,color={'0' : (0,256,0)},parts = f'{args.parts}')
        break

if __name__ == '__main__':
    main()