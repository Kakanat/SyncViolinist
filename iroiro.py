
import numpy as np
import joblib
from QuarterNet.skeleton import Skeleton as Skeleton

def joints2parent(joints:list()):
    sklsource = joblib.load('./data/skl_hiroki_ver/task_2.jb')
    skl = Skeleton(sklsource[0].astype(np.float32), sklsource[1])
    p = []
    idx = 0
    for j in joints:
        if skl.parents()[j] not in joints:
            break
    p.append(j)
    for j in joints[::-1]:
        # print(j)
        # print(joints[::-1])
        if skl.parents()[j] not in joints:
            idx = joints.index(j)
            break
    if j == p[0]:
        return j
    else:
        p = [p[0]]*((idx)) + [j]*(len(joints)-idx)
        # print(len(joints))
        # print(len(p))
        return p


def joints_num(parts:str):
    if parts == 'left_hand':
        return [value for value in range(33,52)]#32かも？
    elif parts == 'right_hand':
        return [value for value in range(9,30)]#8かも？
    elif parts == 'right_hand2':
        return [value for value in range(11,30)]
    elif parts == 'right_arm':  
        return [value for value in range(9,12)]
    elif parts == 'both_hands':
        return [value for value in list(range(9, 30)) + list(range(33, 52))]
    elif parts == 'full_body':
        return [value for value in range(0,62)]
    elif parts == 'woboth_hands':
        return [value for value in list(range(0,10))+list(range(30,34))+list(range(52,62))]
    elif parts == 'left_hand2':
        return [value for value in range(32,52)]#肘スタート
    elif parts == 'woboth_hands2':
        return [value for value in list(range(0,10))+list(range(30,33))+list(range(52,62))]
    elif parts == 'left_hand3':
        return [value for value in range(31,52)]#肩スタート
    elif parts == 'woboth_hands3':
        return [value for value in list(range(0,10))+list(range(30,32))+list(range(52,62))]
    elif parts == 'left_arm':
        return [value for value in range(31,34)]
    elif parts == 'right_leg':
        return [value for value in range(52,57)]
    elif parts == 'left_leg':
        return [value for value in range(57,62)]
    elif parts == 'previs':
        return [52,57,0,1]
    elif parts == 'spine':
        return [value for value in list(range(1,10))+list(range(30,32))]
    elif parts == 'right_hand_left_arm':  
        return [value for value in range(9,30)] + [value for value in range(31,34)]
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
    
def joints_num_hiroki(parts:str):
    if parts == 'left_hand':
        return [value for value in range(39,63)]#32かも？
    elif parts == 'right_hand':
        return [value for value in range(10,36)]#8かも？
    elif parts == 'right_hand_wof':
        return [value for value in list(range(10,36)) if value not in [16,21,26,31,35]]
    elif parts == 'right_hand2':
        return [value for value in range(12,36)]
    elif parts == 'right_arm':  
        return [value for value in range(10,13)]
    elif parts == 'both_hands':
        return [value for value in list(range(10, 36)) + list(range(39, 63))]
    elif parts == 'full_body':
        return [value for value in range(0,75)]
    elif parts == 'woboth_hands':
        return [value for value in list(range(0,11))+list(range(36,40))+list(range(63,75))]
    elif parts == 'left_hand2':
        return [value for value in range(38,63)]#肘スタート
    elif parts == 'woboth_hands2':
        return [value for value in list(range(0,11))+list(range(36,39))+list(range(63,75))]
    elif parts == 'left_hand3':
        return [value for value in range(37,63)]#肩スタート
    elif parts == 'woboth_hands3':
        return [value for value in list(range(0,11))+list(range(36,38))+list(range(63,75))]
    elif parts == 'left_arm':
        return [value for value in range(37,40)]
    elif parts == 'right_leg':
        return [value for value in range(63,69)]
    elif parts == 'left_leg':
        return [value for value in range(69,75)]
    elif parts == 'previs':
        return [63,69,0,1]
    elif parts == 'spine':
        return [value for value in list(range(1,11))+list(range(36,38))]
    elif parts == 'TGM2B':
        return sorted(list(range(0,13)) + list(range(36,40)) + list(range(63,75)))
    elif parts == 'A2BD':
        return sorted(list(range(9,63)))
    elif parts == 'TVCG':
        return sorted(list(0,63))
    elif parts == 'right_hand_left_arm':  
        return [value for value in range(10,36)] + [value for value in range(37,40)]
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
    
    
def joints_tgm2b(parts:str):
    if parts == 'left_hand':
        return [10,11]#32かも？
    elif parts == 'right_hand':
        return [ 13, 14]#8かも？
    elif parts == 'other':
        return [0,1,2,3,4,5,6,7,8,9,12]
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
    
def tgm2bparent(parts:str):
    if parts == 'left_hand':
        return 9
    elif parts == 'right_hand':
        return 12
    elif parts == 'other':
        return 0
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
# ----for debug----
# print(joints_num_hiroki('right_hand_left_arm'))
# # print(joints2parent(joints_num_hiroki('right_hand')))
# print(joints2parent(joints_num_hiroki('right_hand_left_arm')))

