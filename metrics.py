import torch
import torch.nn.functional as F
import numpy as np
from soft_dtw_cuda import SoftDTW
import time
from sklearn.metrics import f1_score
from QuarterNet.skeleton import Skeleton as Skeleton
import torch.nn as nn
"""
based on https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/metric.py
"""

# METRICS = {'L1':[], 'L1RH':[], 'L1LH':[], 'L1LF':[], 'PCK':[], 'DTW':[], 'LCS':[], 'LCS':[],
#          'CosSim':[], 'BowX':[], 'BowY':[], 'BowZ':[], 'BowAve':[]}
# METRICS = {'L1':[], 'L1RH':[], 'L1LH':[], 'L1LF':[], 'DTW':[], 'DTWRH':[], 'DTWLH':[], 'DTWLF':[],
#         'BowX':[], 'BowY':[], 'BowZ':[], 'BowAve':[]}

METRICS = {'L1':[], 'L1RH':[], 'L1LH':[], 'L1LF':[],'L1LF_full':[], 'DTW':[], 'DTWRH':[], 'DTWLH':[], 'DTWLF':[],'DTWLF_full':[],
        'BowX':[], 'BowY':[], 'BowZ':[], 'BowAve':[]}

# ---compute all metrics ---

# def all_mets(pred, targ, mask,loss_fn):
#     if loss_fn == l1l1_loss or l1l2_loss:
#         loss_name = 'L1'
#     elif loss_fn == l2l1_loss or l2l2_loss:
#         loss_name = 'L2'
#     else:
#         raise Exception('Wrong loss function')

#     metrics = METRICS    
#     # L1 losses
#     metrics[f'{loss_name}'] = loss_fn(pred, targ, mask).item()
#     metrics[f'{loss_name}RH'] = loss_fn(pred, targ, mask, [9,10]).item()
#     metrics[f'{loss_name}LH'] = loss_fn(pred, targ, mask, [31,32]).item()
#     metrics[f'{loss_name}LF'] = loss_fn(pred, targ, mask, [36,40,44,48,51]).item()
#     #metrics['PCK'] = compute_pck(pred, targ, mask)
#     # All body 
#     metrics['DTW'] = dtw_tensor(pred, targ, mask)
#     metrics['DTWRH'] = dtw_tensor(pred, targ, mask, [9,10])
#     metrics['DTWLH'] = dtw_tensor(pred, targ, mask, [31,32])
#     metrics['DTWLF'] = dtw_tensor(pred, targ, mask, [36,40,44,48,51])
#     #metrics['LCS'] = longest_common_subsequence_similarity()
#     # Bowing metrics
#     #metrics['Cos sim'] = cosine_similarity(pred, targ, mask)
#     bowing_fmeasure = bowing_acc(pred, targ)
#     metrics['BowX'] = bowing_fmeasure[0]
#     metrics['BowY'] = bowing_fmeasure[1]
#     metrics['BowZ'] = bowing_fmeasure[2]
#     metrics['BowAve'] = bowing_fmeasure[3]
#     return metrics

def all_mets(pred, targ, mask,loss_fn):
    # import IPython;IPython.embed();exit()
    if loss_fn == l1l1_loss or l1l2_loss:
        loss_name = 'L1'
    elif loss_fn == l2l1_loss or l2l2_loss:
        loss_name = 'L2'
    else:
        raise Exception('Wrong loss function')

    metrics = METRICS    
    # L1 losses
    pred = pred.view(-1,75,3)
    targ = targ.view(-1,75,3)
    # mask = mask.view(-1,75,3)

    metrics[f'{loss_name}'] = loss_fn(pred, targ, mask).item()
    metrics[f'{loss_name}RH'] = loss_fn(pred, targ, mask, [11,12]).item()#腕 12も測った方が良き
    metrics[f'{loss_name}LH'] = loss_fn(pred, targ, mask, [38,39]).item()#腕 39も測った方が良き
    metrics[f'{loss_name}LF'] = loss_fn(pred, targ, mask, [43,48,53,58,62]).item()#指先
    metrics[f'{loss_name}LF_full'] = loss_fn(pred, targ, mask, [value for value in range(39,63)]).item()#指先

    #metrics['PCK'] = compute_pck(pred, targ, mask)
    # All body 
    pred = pred.view(1,-1,225)
    targ = targ.view(1,-1,225)
    # mask = mask.view(1,-1,225)
    # import IPython;IPython.embed();exit()
    metrics['DTW'] = dtw_tensor(pred, targ, mask)
    metrics['DTWRH'] = dtw_tensor(pred, targ, mask, [11,12])
    metrics['DTWLH'] = dtw_tensor(pred, targ, mask, [38,39])
    metrics['DTWLF'] = dtw_tensor(pred, targ, mask, [43,48,53,58,62])
    metrics['DTWLF_full'] = dtw_tensor(pred, targ, mask, [value for value in range(39,63)])
    
    # import IPython;IPython.embed();exit()
    #metrics['LCS'] = longest_common_subsequence_similarity()
    # Bowing metrics
    #metrics['Cos sim'] = cosine_similarity(pred, targ, mask)
    bowing_fmeasure = bowing_acc(pred, targ)
    metrics['BowX'] = bowing_fmeasure[0]
    metrics['BowY'] = bowing_fmeasure[1]
    metrics['BowZ'] = bowing_fmeasure[2]
    metrics['BowAve'] = bowing_fmeasure[3]
    return metrics

def all_mets_tgm2b(pred, targ, mask,loss_fn):
    # import IPython;IPython.embed();exit()
    if loss_fn == l1l1_loss or l1l2_loss:
        loss_name = 'L1'
    elif loss_fn == l2l1_loss or l2l2_loss:
        loss_name = 'L2'
    else:
        raise Exception('Wrong loss function')

    metrics = METRICS    
    # L1 losses
    pred = pred.view(-1,75,3)
    targ = targ.view(-1,75,3)
    # mask = mask.view(-1,75,3)

    metrics[f'{loss_name}'] = loss_fn(pred, targ, mask).item()
    metrics[f'{loss_name}RH'] = loss_fn(pred, targ, mask, [13,14]).item()#腕 12も測った方が良き
    metrics[f'{loss_name}LH'] = loss_fn(pred, targ, mask, [10,11]).item()#腕 39も測った方が良き

    #metrics['PCK'] = compute_pck(pred, targ, mask)
    # All body 
    pred = pred.view(1,-1,225)
    targ = targ.view(1,-1,225)
    # mask = mask.view(1,-1,225)
    metrics['DTW'] = dtw_tensor(pred, targ, mask)
    metrics['DTWRH'] = dtw_tensor(pred, targ, mask, [13,14])
    metrics['DTWLH'] = dtw_tensor(pred, targ, mask, [10,11])
    
    return metrics

# --- compute l1 loss ---

def loss_md(pred, targ, mask = 1, joint_ind = None, velocity=False,dims = None):
    if joint_ind == None:
        diff = pred - targ
        # import IPython;IPython.embed();exit()
        loss = torch.norm(diff,p=dims[0],dim=2, keepdim=False, out=None, dtype=None)
    else:
        diff = pred[:,:,joint_ind]-targ[:,:,joint_ind]
        # import IPython;IPython.embed();exit()
        
        loss = torch.norm(diff,p=dims[0],dim=2, keepdim=False, out=None, dtype=None)
    if velocity == True:
        vel_p = torch.abs(pred[:,:-1] - pred[:,1:])
        vel_t = torch.abs(targ[:,:-1] - targ[:,1:])
        diff_vel = vel_p - vel_t
        diff_vel = F.pad(diff_vel, (0,0,1,0))
        loss = torch.norm(diff+diff_vel,p=dims[1],dim=2, keepdim=False, out=None, dtype=None)

    masked_loss = loss * mask
    loss = torch.mean(masked_loss)
    loss = torch.mean(loss)
    
    return loss

def bone_loss(pred, targ, joints, skl):
    # import IPython;IPython.embed();exit()
    
    pred = pred.view(pred.shape[0],pred.shape[1],len(joints),3)
    targ = targ.view(targ.shape[0],targ.shape[1],len(joints),3)
    
    loss = 0
    for j in range(len(joints)):
        p = skl.parents()[joints[j]]
        if p in joints:
            p = joints.index(p)
            b_pred = torch.abs(pred[:,:,j,0] - pred[:,:,p,0]) + torch.abs(pred[:,:,j,1] - pred[:,:,p,1]) + torch.abs(pred[:,:,j,2] - pred[:,:,p,2])
            b_targ = torch.mean(torch.abs(targ[:,:,j,0] - targ[:,:,p,0]) + torch.abs(targ[:,:,j,1] - targ[:,:,p,1]) + torch.abs(targ[:,:,j,2] - targ[:,:,p,2]),dim=1)
            for f in range(b_pred.shape[0]):
                loss += torch.mean(torch.abs(b_pred[:,f] - b_targ[:]))
    return loss


def l1l1_loss(pred, targ, mask=1, joint_ind=None, velocity=False):
    # import IPython; IPython.embed(); exit()
    # raise Exception(pred.shape,pred.max(),targ.max(),pred.min(),targ.min())
    if joint_ind == None:
        diff = torch.abs(pred - targ)
        loss = torch.sum(diff, 2, keepdim=True)
        loss = torch.sum(diff, 1, keepdim=True)
        
    else:
        diff = torch.abs(pred[:,joint_ind,:]-targ[:,joint_ind,:])
        loss = torch.sum(diff, 2, keepdim=True)
        loss = torch.sum(diff, 1, keepdim=True)
        
    if velocity>0:
        vel_p = torch.abs(pred[:,:-1] - pred[:,1:])
        vel_t = torch.abs(targ[:,:-1] - targ[:,1:])
        diff_vel = torch.abs(vel_p - vel_t)
        diff_vel = F.pad(diff_vel, (0,0,1,0))
        loss = torch.sum(diff+diff_vel*velocity, 2, keepdim=True)
    # add up for all keypoints in each time frame
    #loss = torch.sum(diff, 2, keepdim=True) / (diff.size()[-1] + 1e-9)
    
    masked_loss = loss * mask
    loss = torch.mean(masked_loss)
    loss = torch.mean(loss)
    
    return loss

def l2l1_loss(pred, targ, mask=1, joint_ind=None, velocity=False):
    if joint_ind == None:
        diff = (pred - targ)**2
        loss = torch.sum(diff, 2, keepdim=True)
    else:
        diff = (pred[:,:,joint_ind]-targ[:,:,joint_ind])**2
        loss = torch.sum(diff, 2, keepdim=True)
    if velocity == True:
        vel_p = torch.abs(pred[:,:-1] - pred[:,1:])
        vel_t = torch.abs(targ[:,:-1] - targ[:,1:])
        diff_vel = torch.abs(vel_p - vel_t)
        diff_vel = F.pad(diff_vel, (0,0,1,0))
        loss = torch.sum(diff+diff_vel, 2, keepdim=True)
    # add up for all keypoints in each time frame
    #loss = torch.sum(diff, 2, keepdim=True) / (diff.size()[-1] + 1e-9)
    
    masked_loss = loss * mask
    loss = torch.mean(masked_loss)
    loss = torch.mean(loss)
    return loss

def l1l2_loss(pred, targ, mask=1, joint_ind=None, velocity=False):
    if joint_ind == None:
        diff = torch.abs(pred - targ)
        loss = torch.sum(diff, 2, keepdim=True)
    else:
        diff = torch.abs(pred[:,:,joint_ind]-targ[:,:,joint_ind])
        loss = torch.sum(diff, 2, keepdim=True)
    if velocity == True:
        vel_p = torch.abs(pred[:,:-1] - pred[:,1:])
        vel_t = torch.abs(targ[:,:-1] - targ[:,1:])
        diff_vel = (vel_p - vel_t)**2
        diff_vel = F.pad(diff_vel, (0,0,1,0))
        loss = torch.sum(diff+diff_vel, 2, keepdim=True)
    # add up for all keypoints in each time frame
    #loss = torch.sum(diff, 2, keepdim=True) / (diff.size()[-1] + 1e-9)
    
    masked_loss = loss * mask
    loss = torch.mean(masked_loss)
    loss = torch.mean(loss)
    return loss

def l2l2_loss(pred, targ, mask=1, joint_ind=None, velocity=False):
    if joint_ind == None:
        diff = (pred - targ)**2
        loss = torch.sum(diff, 2, keepdim=True)
    else:
        diff = (pred[:,:,joint_ind]-targ[:,:,joint_ind])**2
        loss = torch.sum(diff, 2, keepdim=True)
    if velocity == True:
        vel_p = torch.abs(pred[:,:-1] - pred[:,1:])
        vel_t = torch.abs(targ[:,:-1] - targ[:,1:])
        diff_vel = (vel_p - vel_t)**2
        diff_vel = F.pad(diff_vel, (0,0,1,0))
        loss = torch.sum(diff+diff_vel, 2, keepdim=True)
    # add up for all keypoints in each time frame
    #loss = torch.sum(diff, 2, keepdim=True) / (diff.size()[-1] + 1e-9)
    
    masked_loss = loss * mask
    loss = torch.mean(masked_loss)
    loss = torch.mean(loss)
    return loss



# ---compute dtw ---

def dtw_tensor(pred, targ, mask, joint_ind=None):
    pred[mask == 0] = 0
    length = pred.shape[1]
    if joint_ind == None:
        _pred = pred
        _targ = targ
        joint_num = 75
    else:
        pred = pred.view(-1,75,3)
        pred = pred[:,joint_ind,:]
        _pred = pred.view(1,-1,3*len(joint_ind))
        targ = targ.view(-1,75,3)
        targ = targ[:,joint_ind,:]
        _targ = targ.view(1,-1,3*len(joint_ind))
        
        # _pred = pred[:,:,joint_ind]
        # _targ = targ[:,:,joint_ind]
        joint_num = len(joint_ind)
    dtw = 0
    #dtw = torch.empty((1,_pred.size()[2]))
    dtw_func = SoftDTW(use_cuda=True, gamma=1) # not softDTW (general DTW)
    for i in range(length // 128):
        dtw += dtw_func(_pred[:,i*128:(i+1)*128], _targ[:,i*128:(i+1)*128]).item()
    return dtw / (length // 128) / 128 

# --- compute cosine_similarity ---

def cosine_similarity(pred, targ, mask):
    cs = 0
    pass

# --- compute LCS ---

def longest_common_subsequence_similarity(pred, targ, mask):
    lcs = 0
    pass

# --- compute PCK ---
def compute_pck(pred, gt, alpha=0.1):
    """
    https://github.com/amirbar/speech2gesture/blob/master/common/evaluation.py
    
    Args:
        pred: predicted keypoints on NxMxK where N is number of samples, M is of shape 2, corresponding to X,Y and K is the number of keypoints to be evaluated on
        gt:  similarly
        lpha: parameters controlling the scale of the region around the image multiplied by the max(H,W) of the person in the image. We follow https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1
    Returns: 
        mean prediction score
    """
    pred = np.reshape(pred, [len(pred), 3, -1])  
    gt = np.reshape(gt, [len(pred), 3, -1])
    pck_radius = compute_pck_radius(gt, alpha)
    keypoint_overlap = (np.linalg.norm(np.transpose(gt-pred, [0, 2, 1]), axis=2) <= (pck_radius))
    return np.mean(keypoint_overlap)

def compute_pck_radius(gt, alpha):
    width = np.abs(np.max(gt[:, 0:1], axis=2) - np.min(gt[:, 0:1], axis=2))
    depth = np.abs(np.max(gt[:, 1:2], axis=2) - np.min(gt[:, 1:2], axis=2))
    height = np.abs(np.max(gt[:, 2:3], axis=2) - np.min(gt[:, 2:3], axis=2))
    max_axis = np.concatenate([width, depth, height], axis=1).max(axis=1)
    max_axis_per_keypoint = np.tile(np.expand_dims(max_axis, -1), [1, 15])
    return max_axis_per_keypoint * alpha

# ---compute fmeasure for bow attack ---
def compute_bowing_attack(direction):
    direction = np.sign(direction) # 1 0 -1
    temp = direction[0]
    bowing_attack = np.zeros_like(direction)
    for i, d in enumerate(direction):
        if i != 0:
            if d != temp:
                bowing_attack[i] = 1
                temp = d
    bowing_attack = bowing_attack.astype(int)
    return bowing_attack

def bowing_acc(pred, targ, alpha=3):
    """
    Args:
        pred, targ: [N, T, (K*3)]
        alpha: tolerance
    Returns:
        F1 score of bowing attack accuracy on x, y, z
    """
    pred = pred.to('cpu').detach().numpy().copy()
    pred = np.squeeze(pred[:,:,33:36] - pred[:,:,0:3])
    targ = targ.to('cpu').detach().numpy().copy()
    targ = np.squeeze(targ[:,:,33:36] - targ[:,:,0:3])

    F1 = []
    pred_direction = pred[1:] - pred[:-1]
    targ_direction = targ[1:] - targ[:-1]
    for coordinate in range(3):      
        pred_bow = compute_bowing_attack(pred_direction[:, coordinate]) 
        targ_bow = compute_bowing_attack(targ_direction[:, coordinate])
        prediction = []
        label = []
        i = 0
        index = np.zeros_like(targ_bow) # record which index has been calculated already
        for p, t in zip(pred_bow, targ_bow):
            if p == 1:
                prediction.append(1)
                if i-alpha < 0:
                    temp = targ_bow[0:i+alpha+1]
                    temp_idx = index[0:i+alpha+1]
                    
                elif i+alpha > len(pred_bow)-1:
                    temp = targ_bow[i-alpha:]
                    temp_idx = index[i-alpha:]
                    
                else:
                    temp = targ_bow[i-alpha:i+alpha+1]
                    temp_idx = index[i-alpha:i+alpha+1]
                    
                if 1 in temp:
                    idx = 0
                    token = 0
                    for t, t_idx in zip(temp, temp_idx):
                        if t == 1 and t_idx == 0:
                            label.append(1)
                            if i-alpha < 0:
                                index[idx] = 1
                                
                            else:
                                index[i-alpha+idx] = 1
                                
                            token = 1
                            break
                        idx += 1
                    if token == 0:
                        label.append(0)
                    
                else:
                    label.append(0)

            elif p==0 and t==1:
                prediction.append(0)
                label.append(1)
                
            elif p==0 and t==0:
                prediction.append(0)
                label.append(0)         
                
            i+=1             
            
        f1 = f1_score(label, prediction) 
        F1.append(f1)
    return (F1[0], F1[1], F1[2], np.mean(F1))

# for debug
def main():
    B = 32
    T = 128
    J = 62 
    joint_ind = [1,2]
    device = torch.device(0)
    dummy_pred = torch.randn((B, T, J*3)).to(device)
    dummy_targ = torch.randn((B, T, J*3)).to(device)
    pred_np = dummy_pred.to('cpu').detach().numpy().copy()
    targ_np = dummy_targ.to('cpu').detach().numpy().copy()
    mask = dummy_targ != 0
    mask_np = targ_np != 0
    #print(l1_loss(dummy_pred, dummy_targ, mask, joint_ind)) # -> ok
    print(dtw_tensor(dummy_pred, dummy_targ, mask, joint_ind)) #-> ok
    #print(compute_pck(pred_np, targ_np), alpha=0.1)
    print()

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, outputs, targets):
        loss = self.bceloss(outputs, targets)
        return loss


class JointRotationLoss(nn.Module):
    def __init__(self):
        super(JointRotationLoss, self).__init__()
        self.l1loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, outputs, targets):
        loss = self.l1loss(outputs, targets)
        return loss

class EndJointPositionLoss(nn.Module):
    def __init__(self):
        super(EndJointPositionLoss, self).__init__()
        self.l1loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, outputs, targets):
        loss = self.l1loss(outputs, targets)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self, w_joint_rotation = 100, w_end_effector = 100):
        self.ganloss = GANLoss()
        self.jrloss = JointRotationLoss()
        self.ejploss = EndJointPositionLoss()

        self.w_jr = w_joint_rotation
        self.w_ejp = w_end_effector

    def forward(self, outputs, targets):
        loss = self.ganloss(outputs, targets) + self.w_jr * self.jrloss(outputs, targets) + self.w_ejp * self.ejploss(outputs, targets)
        return loss

if __name__ == '__main__':
    main()