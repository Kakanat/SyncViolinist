import torch 
# from models.LightningModel import LitLDA
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import yaml
import glob
import joblib
import random
import os 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pathlib 
import sys
from metrics import l1l1_loss,l1l2_loss,l2l1_loss,l2l2_loss,loss_md,bone_loss,GANLoss, JointRotationLoss, EndJointPositionLoss
from iroiro import joints2parent,joints_num_hiroki,joints_num
import datetime
from tqdm import tqdm
from model import Model,FBNet,Generator
import metrics as mts
from data import KeypsPPDataset, Normalize, woPPDataset,ViolinMotionDataset,ViolinMotionDataset_FullLength,woPPkeypsDataset
from utils import sort_seq, save_video
sys.path.append('../')
from QuarterNet import skeleton
from QuarterNet.skeleton import Skeleton as Skeleton
from QuarterNet import visualization as vis 


import datetime
PROPOSE = True
WANDB = False
METRICS = {'L1':[], 'L1RH':[], 'L1LH':[], 'L1LF':[],'L1LF_full':[], 'DTW':[], 'DTWRH':[], 'DTWLH':[], 'DTWLF':[],'DTWLF_full':[],
        'BowX':[], 'BowY':[], 'BowZ':[], 'BowAve':[]}
vis_gt = False
vis_parts = False
check_train_data= False

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def oh(pred: torch.tensor):
    cls = pred.shape[1]
    pred = torch.argmax(pred,dim=1)
    onehot = torch.zeros([pred.shape[1],cls]).to(pred.device)
    for i in range(pred.shape[1]):
        onehot[i,pred[0,i]] = 1
    return onehot

def transform_keyps(y, mean, std):
    y = std * y + mean
    return y

def test(net_f1,net_f2,net_f3,net_f4,net_b, device, test_loader,filename,joints,dims,skl,loss_fn=l1l1_loss,oracle=False,pp=None):
    if not oracle:
        net_f1.eval()
        net_f2.eval()
        net_f3.eval()
        net_f4.eval()
        net_b.eval()
    test_loss = 0.0
    test_loss_l1 = 0.0
    test_loss_l2 = 0.0
    bl = 0.0
    all_preds = []
    all_gts = []
    mets = METRICS
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            if 'TF' in net_b.name():
                fs = 128
                frame = data[3] if oracle else data[2]
            else:
                fs = int(data[3]) if oracle else int(data[2])
                frame = data[3] if oracle else data[2]
            preds = []
            gts = []
            for f in range(0,frame,fs):
                if oracle:
                    x, x2, targ, seq_length = data[0][:,f:f+fs,:].to(device), data[1][:,f:f+fs,:].to(device), data[2][:,f:f+fs,:].to(device), torch.tensor([fs])
                    x,lengths = sort_seq(x, seq_length)
                    
                else:
                    x, targ, seq_length = data[0][:,f:f+fs,:].to(device), data[1][:,f:f+fs,:].to(device), torch.tensor([fs])
                    x,lengths = sort_seq(x, seq_length)
                    if pp == '1111':
                        x2 = torch.cat([oh(net_f1(x,lengths)[0]),oh(net_f2(x,lengths)[1]),oh(net_f3(x,lengths)[2]),oh(net_f4(x,lengths)[3])],dim=1).unsqueeze(0)
                    elif pp == '0111':
                        x2 = torch.cat([oh(net_f2(x,lengths)[1]),oh(net_f3(x,lengths)[2]),oh(net_f4(x,lengths)[3])],dim=1).unsqueeze(0)
                    elif pp == '1011':
                        x2 = torch.cat([oh(net_f1(x,lengths)[0]),oh(net_f3(x,lengths)[2]),oh(net_f4(x,lengths)[3])],dim=1).unsqueeze(0)
                    elif pp == '1101':
                        x2 = torch.cat([oh(net_f1(x,lengths)[0]),oh(net_f2(x,lengths)[1]),oh(net_f4(x,lengths)[3])],dim=1).unsqueeze(0)
                    elif pp == '1110':
                        x2 = torch.cat([oh(net_f1(x,lengths)[0]),oh(net_f2(x,lengths)[1]),oh(net_f3(x,lengths)[2])],dim=1).unsqueeze(0)
                        
                    x2 = sort_seq(x2, seq_length)[0]
                    
                    
                    
                gt = targ
                targ_r = torch.reshape(targ,(-1,75,3))[:,joints,:]
                targ = torch.reshape(targ_r,(targ.shape[0], targ.shape[1], len(joints)*3))
                
                targ = sort_seq(targ, seq_length)[0]
                pred = net_b(x, x2, sort_seq(x, seq_length)[1])
                
                if dims:
                    loss = loss_fn(pred, targ, mask=1,dims=dims)
                else:
                    loss = loss_fn(pred, targ, mask=1)
                lossl1 = l1l1_loss(pred,targ,mask=1,velocity=False)
                lossl2 = l2l2_loss(pred,targ,mask=1,velocity=False)

                bl += bone_loss(pred, targ, joints, skl)
                test_loss += loss.item() 
                test_loss_l1 += lossl1.item()
                test_loss_l2 += lossl2.item()

                all_preds.append(pred)
                all_gts.append(gt)
    
    preds = torch.squeeze(torch.cat(all_preds,dim=1).reshape(len(all_preds), *all_preds[0].shape))
    preds = torch.reshape(preds, (-1, len(joints), 3))
    
    gts = torch.squeeze(torch.cat(all_gts,dim=1).reshape(len(all_gts), *all_gts[0].shape))
    gts = torch.reshape(gts, (-1, 75, 3))
    print(gts.shape)
    print(f'Test: Loss: {test_loss/len(test_loader)}')
    print(f'Test: Loss_l1: {test_loss_l1/len(test_loader)}')
    print(f'Test: Loss_l2: {test_loss_l2/len(test_loader)}')
    print(f'Test: bone loss: {bl/len(test_loader)}')
    
    return preds, mets,gts,test_loss_l1/len(test_loader), bl/len(test_loader)

def test_G(G, device, test_loader, skl, l_jr, l_ejp, l_gan):
    test_loss_G  = 0.0
    pred_qs = np.empty((0,62,4), dtype=np.float32)
    G.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            bsize = len(data[0])
            frame = data[0].shape[2]
            x, targ_q, targ_p = data[0].to(device), data[1].to(device), data[2].reshape((bsize, frame,-1)).transpose(2,1).to(device)
            
            pred_q = G(x)
            pred_q2 = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(bsize, frame, 75, 1).to(device)
            targ_q2 = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(bsize, frame, 75, 1).to(device)
            pred_q2[:,:,list(set(skl.parents()[1:])),:] = pred_q.clone()
            targ_q2[:,:,list(set(skl.parents()[1:])),:] = targ_q.clone()
            rp = torch.zeros((bsize, frame, 3)).to(device)
            pred_p = skl.forward_kinematics(pred_q2, rp)
            pred_p = pred_p.reshape((bsize, frame, -1)).permute((0,2,1))
            pred_keyp = skl.forward_kinematics(pred_q2, rp)
            pred_keyps = pred_keyp
            rp = torch.zeros((bsize, frame, 3)).to(device)
            targ_p = skl.forward_kinematics(targ_q2, rp)
            targ_p = targ_p.reshape((bsize, frame, -1)).permute((0,2,1))
            targ_keyp = skl.forward_kinematics(targ_q2, rp)
            targ_keyps = targ_keyp
            fake_feats = torch.cat((x, pred_q.reshape((bsize, frame, -1)).transpose(2,1)), dim=1)
            fake_targs = torch.ones([fake_feats.shape[0], 1]).to(device)
            loss_G = l_jr(pred_q, targ_q) + l_ejp(pred_p, targ_p)
            test_loss_G += loss_G.item()

            pred_qs = np.append(pred_qs, pred_q.to('cpu').detach().numpy().copy().reshape((bsize*frame, -1, 4)), axis=0)

        print('test loss (Generator): ', test_loss_G / len(test_loader))
    return pred_qs, pred_keyps.reshape((-1,75,3)), targ_keyps.reshape((-1,75,3))




def create_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='A2BD')
    parser.add_argument('--visualize', action = 'store_true')
    parser.add_argument('--oracle', action = 'store_true')
    parser.add_argument('--o', type=str, default = './results')
    parser.add_argument('--gpu', type=str, default = '0')
        
    args = parser.parse_args()
    return args


def main():
    today = datetime.date.today()
    seed_everything()
    args = create_options()
    if not os.path.exists("results"):
        os.mkdir("results")
    metrics_dict = METRICS
    config = {}
    wav_dir = './data/wav_normalized/'

    device=f'cuda:{args.gpu}'
    
    if check_train_data:
        aud_data = sorted(glob.glob(args.aud + '/train/*.jb'))
        joint_data = sorted(glob.glob(args.keyps + '/train/*.jb'))
        pp_data = sorted(glob.glob("1111" + '/train/*.jb'))

    model_pth = {}
    dims = None
    model_dir={}
    if 'TVCG' in args.model or 'hirata' in args.model or 'hiroki' in args.model or 'LDA' in args.model:
        aud_data = sorted(glob.glob('./data/aud/test/*.jb'))
    else:
        aud_data = sorted(glob.glob('./data/mfcc/test/*.jb'))
        
    if 'TVCG' not in args.model:
        joint_data = sorted(glob.glob('./data/keyps_norm_hiroki_ver_6ch/test/*.jb'))
    else:
        joint_data = sorted(glob.glob('./data/joint_aligned/test/*.jb'))
        
    if args.oracle:
        pp_data = sorted(glob.glob('./data/pp_one_hot/test/*.jb'))
    else:
        pp_data=[None]*8
        
    if args.model == 'hiroki_tune5':
        model_dir[f'{0}'] = f'./models/hiroki/hiroki-woboth_hands3_128_3_1111_1/'
        model_pth[f'{0}'] = f'./models/hiroki/hiroki-woboth_hands3_128_3_1111_1/' + 'best.pth'
        with open(model_dir[f'{0}']+'config.yaml') as f:
            config[f'{0}'] = yaml.safe_load(f)
        model_dir[f'{1}'] = f'./models/hiroki/hiroki-left_arm_256_2_1111_1/'
        model_pth[f'{1}'] = f'./models/hiroki/hiroki-left_arm_256_2_1111_1/' + 'best.pth'
        with open(model_dir[f'{1}']+'config.yaml') as f:
            config[f'{1}'] = yaml.safe_load(f)
        
        model_dir[f'{2}'] = f'./models/hiroki/hiroki-left_hand_256_2_1111_1/'
        model_pth[f'{2}'] = f'./models/hiroki/hiroki-left_hand_256_2_1111_1/' + 'best.pth'
        with open(model_dir[f'{2}']+'config.yaml') as f:
            config[f'{2}'] = yaml.safe_load(f)
        model_dir[f'{3}'] = f'./models/hiroki/hiroki-right_hand_512_2_1111_1/'
        model_pth[f'{3}'] = f'./models/hiroki/hiroki-right_hand_512_2_1111_1/' + 'best.pth'
        with open(model_dir[f'{3}']+'config.yaml') as f:
            config[f'{3}'] = yaml.safe_load(f)
            
            
    elif args.model == 'A2BD':
        model_dir[f'{0}'] = 'models/A2BD/'
        model_pth[f'{0}'] = 'models/A2BD/' + 'best.pth'
        with open(model_dir[f'{0}']+'config.yaml') as f:
            config[f'{0}'] = yaml.safe_load(f)
            
    elif args.model == 'TGM2B_fullbody_rhoff':
        model_dir[f'{0}'] = 'models/TGM2B/'        
        model_pth[f'{0}'] = 'models/TGM2B/' + 'best.pth'
        with open(model_dir[f'{0}']+'config.yaml') as f:
            config[f'{0}'] = yaml.safe_load(f)
            
    elif args.model == 'TVCG':
        model_pth['0'] = 'models/TVCG/best.pth'
        with open('models/TVCG/config.yaml') as f:
            config[f'{0}'] = yaml.safe_load(f)
    
    skl_file = "./data/skl_hiroki_ver/task_2.jb"
    sklsource = joblib.load(skl_file)
    skl = Skeleton(sklsource[0].astype(np.float32), sklsource[1])
    
    if 'TVCG' not in args.model:
        pp_all = ('string', 'fing', 'pos', 'ud')
        pp_all_dim = (4, 5, 12, 2)
        pp = [pp_all[i] for i in range(4) if int(config['0']['pp_mode'][i])==1] 
        pp_dim = sum([pp_all_dim[i] for i in range(4) if int(config['0']['pp_mode'][i])==1])


        print(aud_data, joint_data, pp_data)
        for aud_d, joint_d, pp_d in tqdm(zip(aud_data, joint_data, pp_data)):   
            
            transform = Normalize()
            
            if args.oracle:
                test_data = KeypsPPDataset([aud_d], [joint_d],[pp_d], pp=pp, transform = transform, test_state = config['0'])
                test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
            elif aud_d=='./data/new_audio.jb':
                test_data = woPPkeypsDataset([aud_d], transform = transform, test_state = config['0'])
                test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)    
            else:
                test_data = woPPDataset([aud_d], [joint_d], transform = transform, test_state = config['0'])
                test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
                
            for data in test_loader:
                if args.oracle:
                    frame = data[3]
                else:
                    frame = data[2]
                    
            pred_keyps = torch.zeros([frame,75,3]).to(device)
            loss = 0
            bone = 0
            
            for i in range(len(config)):
                joints = joints_num_hiroki(config[f'{i}']['parts'])
                parent = joints2parent(joints)
                if 'TGM2B' not in config[f'{i}']['model']:
                    config[f'{i}']['params']['output_dim'] = len(joints)*3
                else:
                    config[f'{i}']['params']['d_output_body'] = len(joints)*3 - 6
                    
                model_pth1 = './models/hiroki/FBestimator/only_ws_tkcrnn_1133/'
                model_pth2 = './models/hiroki/FBestimator/only_wf_tkcrnn_1129/'
                model_pth3 = './models/hiroki/FBestimator/only_wp_tkcrnn_1122/'
                model_pth4 = './models/hiroki/FBestimator/only_wu_tkcrnn_1118/'

                with open(model_pth1+'config.yaml') as f:
                    config1 = yaml.safe_load(f) 
                with open(model_pth2+'config.yaml') as f:
                    config2 = yaml.safe_load(f) 
                with open(model_pth3+'config.yaml') as f:
                    config3 = yaml.safe_load(f) 
                with open(model_pth4+'config.yaml') as f:
                    config4 = yaml.safe_load(f) 
                
                if not args.oracle:
                    net_f1 = FBNet(config1)
                    net_f1 = net_f1.to(device)
                    net_f1.load_state_dict(torch.load(model_pth1+'FBNet_best.pth'))

                    net_f2 = FBNet(config2)
                    net_f2 = net_f2.to(device)
                    net_f2.load_state_dict(torch.load(model_pth2+'FBNet_best.pth'))

                    net_f3 = FBNet(config3)
                    net_f3 = net_f3.to(device)
                    net_f3.load_state_dict(torch.load(model_pth3+'FBNet_best.pth'))

                    net_f4 =FBNet(config4)
                    net_f4 = net_f4.to(device)
                    net_f4.load_state_dict(torch.load(model_pth4+'FBNet_best.pth'))
                    
                    
                else:
                    net_f1,net_f2,net_f3,net_f4=None,None,None,None
                    
                if 'diff' in config[f'{i}']['model']:
                    raise Exception('not implemented')
                    net_b = LitLDA(config[f'{i}'])
                else:
                    net_b = Model(config[f'{i}']['model'], config[f'{i}']['params'])
                net_b = net_b.to(device)
                net_b.load_state_dict(torch.load(model_pth[f'{i}'], map_location=device))
                print(aud_d)
                
                pred, met ,gt,l1,bl= test(net_f1,net_f2,net_f3,net_f4,net_b, device, test_loader,aud_d,joints,dims=dims,skl=skl,loss_fn=l1l1_loss,oracle=args.oracle,pp = "1111")
                loss+=l1
                bone+=bl
                
                print(pred.shape)
                if parent!=0:
                    print('parent is not 0')
                    for k in range(pred.shape[1]):
                        if type(parent) != list:
                            pred[:,k,:] = pred[:,k,:]+pred_keyps[:,parent,:]
                        else:
                            pred[:,k,:] = pred[:,k,:]+pred_keyps[:,parent[k],:]
                            
                pred_keyps[:,joints,:] = pred
            
            print(f'l1 loss total : {loss}')
            print(f'bone loss total : {bone}')
            
            out_dir = args.o
            if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
            if 'TGM2B' in args.model:
                if 'rhoff' in args.model:
                    pred_keyps[:,[11,12],:]  = pred_keyps[:,[11,12],:]+pred_keyps[:,[10],:]
                if not 'fullbody' in args.model:
                    pred_keyps[:,joints_num_hiroki('left_hand'),:] = gt[0,joints_num_hiroki('left_hand'),:]-gt[0,[39],:]+pred_keyps[:,[39],:]
                    pred_keyps[:,joints_num_hiroki('right_hand2'),:] = gt[0,joints_num_hiroki('right_hand2'),:]-gt[0,[12],:]+pred_keyps[:,[12],:]
                else:
                    pred_keyps[:,joints_num_hiroki('right_hand2')[1:],:] = pred_keyps[:,joints_num_hiroki('right_hand2')[1:],:] + pred_keyps[:,[12],:] - pred_keyps[:,[10],:]
                    
            if args.model == 'A2BD_w_gt':
                pred_keyps[:,list(range(0,9)),:] = gt[0,list(range(0,9)),:]
                pred_keyps[:,list(range(63,75)),:] = gt[0,list(range(63,75)),:]
                pred_keyps[:,list(range(9,36)),:] = pred_keyps[:,list(range(9,36)),:] - pred_keyps[:,[9],:] + gt[0,[9],:]
                pred_keyps[:,list(range(36,63)),:] = pred_keyps[:,list(range(36,63)),:] - pred_keyps[:,[36],:] + gt[0,[36],:]
                
            met = mts.all_mets(pred_keyps.view(1,-1,225), gt.view(1,-1,225), mask=1,loss_fn = l1l1_loss)
            for k in metrics_dict:
                metrics_dict[k].append(met[k])
            
            pred_keyps = (pred_keyps - pred_keyps.mean())/pred_keyps.std()
            pred_keyps = (pred_keyps+skl.offsets().mean().item())*skl.offsets().std().item()
            if PROPOSE:
                np.save(f'{out_dir}/gt_new_{Path(aud_d).stem}.npy', gt.to('cpu'))
                np.save(f'{out_dir}/proposal_new_{Path(aud_d).stem}.npy', pred_keyps.to('cpu'))
                
            if args.visualize:
                audiofile = wav_dir + pathlib.Path(aud_d).stem + '.wav'
                
                videofile = out_dir+'/' + pathlib.Path(aud_d).stem + '.mp4'
                joints = joints_num_hiroki('full_body')
                vis.render_joint_cv2(pred_keyps[:,joints,:].to('cpu').detach().numpy().copy(), skl, 30.0, videofile, audiopath=audiofile)#こっち最新
    if 'TVCG' in args.model:
        
        metrics_dict = METRICS
        e = 0
        for i in aud_data:
            s=e
            e+=(joblib.load(i).shape[1]//128 - 2)*128
            
            transform = Normalize()
            
            test_data = ViolinMotionDataset_FullLength([i], [joint_data[aud_data.index(i)]], joint_data[0], transform = transform, test_state = {'aud_mean':config['0']['aud_mean'], 'aud_std':config['0']['aud_std']})
            
            test_loader_b1 = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            test_loader_b32 = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
            test_loader_b128 = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
            
            G = Generator(config['0']).to(device)
            
            G.load_state_dict(torch.load(model_pth['0'], map_location=device))
            D = 0
            
            l_gan = GANLoss()
            l_jr = JointRotationLoss()
            l_ejp = EndJointPositionLoss()

            sklsource = joblib.load('./data/skl_hiroki_ver/task_2.jb')
            skl = Skeleton(sklsource[0].astype(np.float32), sklsource[1])
            pred_q, pred_keyps, targ_keyps = test_G(G, device, test_loader_b32, skl, l_jr, l_ejp, l_gan)
            
            out_dir = args.o
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            pred=pred_keyps
            gt = targ_keyps
            mask = gt != 0
            pred = (pred - pred.mean())/pred.std()
            gt = (gt - gt.mean())/gt.std()
            met = mts.all_mets(pred.reshape([-1,225]).unsqueeze(0), gt.reshape([-1,225]).unsqueeze(0), mask=1,loss_fn = l1l1_loss)
            
            for k in metrics_dict:
                metrics_dict[k].append(met[k])    
            
            videofile = os.path.join(out_dir, Path(i).stem + '.mp4')
            videofile_gt = os.path.join(out_dir , 'gt' , Path(i).stem + '.mp4')
            print(videofile)
            if args.visualize:
                pred = pred.to('cpu').detach().numpy().copy()
                targ = gt.to('cpu').detach().numpy().copy()
                
                off = (pred[:,[66],:]+pred[:,[72],:])/2
                
                pred = pred - off
                
                pred = (pred - pred.mean())/pred.std()
                pred = (pred+skl.offsets().mean().item())*skl.offsets().std().item()
                
                np.save(f'{out_dir}/gt_new_{Path(i).stem}.npy', targ)
                np.save(f'{out_dir}/proposal_new_{Path(i).stem}.npy', pred)
                
                vis.render_joint_cv2(pred, skl, 30.0,videofile,f'./data/wav_normalized/{Path(i).stem}.wav')
                vis.render_joint_cv2(targ, skl, 30.0,videofile_gt,f'./data/wav_normalized/{Path(i).stem}.wav')
                

    metrics_dict['model'] = [args.model]*8
    metrics_dict['configs'] = [[Path(value).stem for value in model_dir.values()]]*8
    if not os.path.exists('./evaluation_results.csv'):
        pd.DataFrame(metrics_dict).to_csv('./evaluation_results.csv',index=False)
    else:
        pd.concat([pd.read_csv('./evaluation_results.csv'),pd.DataFrame(metrics_dict)],axis=0).to_csv('./evaluation_results.csv',index=False)

if __name__ == '__main__':
    main()
