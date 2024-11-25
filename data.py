import numpy as np
import torch 
import torchvision.transforms as transforms
import joblib


class ViolinMotionDataset(torch.utils.data.Dataset):
    def __init__(self, audfiles, jointfiles, sklfile, transform=None, test_state=None):

        self.transform = transform
        self.test_state = test_state

        self.aud = []
        self.rot = []
        self.ep = []
        self.rp = []

        self.datanum = 0

        self.aud_mean = 0
        self.aud_std = 0
        # self.pos_mean = 0
        # self.pos_std = 0

        # --- load audio data ---
        for file in audfiles:
            if len(file) > 0:
                all_aud = joblib.load(file)
                for i in range(all_aud.shape[1]//128 - 2):
                    self.aud.append(all_aud[:, 128*i:128*(i+1)])
                    self.datanum += 1

        # --- load skl data ---
        sklinfo = joblib.load(sklfile)
        self.offsets = sklinfo[0].astype(np.float32)
        self.parents = sklinfo[1]

        # --- load joint data ---
        for file in jointfiles:
            if len(file) > 0:
                all_joint = joblib.load(file)
                for i in range(all_joint[2].shape[0]//128 - 2):
                    self.rot.append(all_joint[2][128*i:128*(i+1)].astype(np.float32))
                    self.ep.append(all_joint[3][128*i:128*(i+1)].astype(np.float32))
                    self.rp.append(all_joint[4][128*i:128*(i+1)].astype(np.float32))

        self.aud_mean, self.aud_std = np.mean(np.array(self.aud)), np.std(np.array(self.aud))
        #self.pos_mean, self.pos_mean = torch.std_mean(self.keyps)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):

        out_aud = self.aud[idx]
        out_rot = self.rot[idx]
        out_ep = self.ep[idx]
        out_rp = self.rp[idx]

        if self.test_state:
            self.aud_mean = self.test_state['aud_mean']
            self.aud_std = self.test_state['aud_std']
            #self.pos_mean = self.test_state['pos_mean']
            #self.pos_std = self.test_state['pos_std']

        if self.transform:
            out_aud = self.transform(out_aud, self.aud_mean, self.aud_std)

            #out_ep = self.transform(out_ep, self.pos_mean, self.pos_std)

        return out_aud, out_rot, out_ep, out_rp

class ViolinMotionDataset_FullLength(torch.utils.data.Dataset):
    def __init__(self, audfiles, jointfiles, sklfile, transform=None, test_state=None):

        self.transform = transform
        self.test_state = test_state

        self.aud = []
        self.rot = []
        self.ep = []
        self.rp = []

        self.datanum = 0

        self.aud_mean = 0
        self.aud_std = 0
        # self.pos_mean = 0
        # self.pos_std = 0

        # --- load audio data ---
        for file in audfiles:
            if len(file) > 0:
                all_aud = joblib.load(file)
                pad = (all_aud.shape[1]//128)*128-all_aud.shape[1]
                if pad>0:
                    self.aud.append(np.pad(all_aud,[(0,0),(0,0),(0,pad)]))
                else:
                    self.aud.append(all_aud[:,:pad])
                self.datanum += 1

        # --- load skl data ---
        sklinfo = joblib.load(sklfile)
        self.offsets = sklinfo[0].astype(np.float32)
        self.parents = sklinfo[1]

        # --- load joint data ---
        for file in jointfiles:
            if len(file) > 0:
                all_joint = joblib.load(file)
                f_diff = all_joint[2].shape[0] - self.aud[0].shape[1]
                if f_diff>=0:
                    self.rot.append(all_joint[2][:-f_diff].astype(np.float32))
                    self.ep.append(all_joint[3][:-f_diff].astype(np.float32))
                    self.rp.append(all_joint[4][:-f_diff].astype(np.float32))
                else:
                    self.rot.append(np.pad(all_joint[2].astype(np.float32),[(0,0),(0,f_diff),(0,0)]))
                    self.ep.append(np.pad(all_joint[3].astype(np.float32),[(0,0),(0,f_diff),(0,0)]))
                    self.rp.append(np.pad(all_joint[4].astype(np.float32),[(0,0),(0,f_diff),(0,0)]))
        self.aud_mean, self.aud_std = np.mean(np.array(self.aud)), np.std(np.array(self.aud))
        #self.pos_mean, self.pos_mean = torch.std_mean(self.keyps)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):

        out_aud = self.aud[idx]
        out_rot = self.rot[idx]
        out_ep = self.ep[idx]
        out_rp = self.rp[idx]

        if self.test_state:
            self.aud_mean = self.test_state['aud_mean']
            self.aud_std = self.test_state['aud_std']
            #self.pos_mean = self.test_state['pos_mean']
            #self.pos_std = self.test_state['pos_std']

        if self.transform:
            out_aud = self.transform(out_aud, self.aud_mean, self.aud_std)

            #out_ep = self.transform(out_ep, self.pos_mean, self.pos_std)

        return out_aud, out_rot, out_ep, out_rp

class KeypsPPDataset(torch.utils.data.Dataset):
    def __init__(self, audfiles, keypsfiles, ppfiles, 
        pp = ['string','fing','pos','ud'], transform=None, val_state=None, test_state=None):
        self.transform = transform
        self.state_for_test = test_state
        self.state_for_val = val_state
        self.aud = []
        self.pp = []
        self.keyps = []
        self.seq = []
        self.data_num = 0
        self.keyp_ind = 0

        self.aud_mean = 0
        self.aud_std = 0
        self.keyps_mean = 0
        self.keyps_std = 0
        self.pp_ind = []
        f_size = 128#128
        pp_ind_string = [0,1,2,3,4]
        pp_ind_fing = [5,6,7,8,9,10]
        pp_ind_pos = [11+i for i in range(13)]
        pp_ind_ud = [24, 25, 26]

        for p in pp:
            ns = locals()
            self.pp_ind += ns['pp_ind_' + p] 
        
        # load audio features
        for f in audfiles:
            if f:
                print(f'data.py KeypsPPDataset : f in audfiles is True')
                # import IPython;IPython.embed();exit()
                aud_all = joblib.load(f)
                aud_all = aud_all.transpose(1, 0)
                for i in range(len(aud_all)//f_size + 1):#元々128
                    x = aud_all[f_size*i:min(len(aud_all), f_size*(i+1))]
                    x = torch.from_numpy(x.astype(np.float32)).clone()
                    self.aud.append(x)
                    self.seq.append(x.size()[0])
                    self.data_num += 1
            else:
                print(f'data.py KeypsPPDataset : f in audfiles is None')
        for f in ppfiles:
            if f:
                print(f'data.py KeypsPPDataset : f in audfiles is True')

                pp_all = joblib.load(f)[list(self.pp_ind)]
                pp_all = pp_all.transpose(1, 0)
                for i in range(len(pp_all)//f_size + 1):
                    x2 = pp_all[f_size*i:min(len(pp_all), f_size*(i+1))]
                    x2 = torch.from_numpy(x2.astype(np.float32)).clone()
                    self.pp.append(x2)
            else:
                print(f'data.py KeypsPPDataset : f in ppfiles is None')
        # load keyps featues
        # import IPython;IPython.embed();exit()
        for f in keypsfiles:
            if f:
                keyps_all = joblib.load(f)
                for i in range(len(keyps_all)//f_size + 1):
                    y = keyps_all[f_size*i:min(len(keyps_all),f_size*(i+1))]
                    y = torch.from_numpy(y.astype(np.float32)).clone()
                    y = y.view((y.size(0),-1))
                    if y.size()[0] != self.aud[self.keyp_ind].size()[0]:
                        import IPython;IPython.embed();exit()
                        print('Audio features and keypoint features have different lengths.')
                        print(self.aud[i].size()[0], y.size())
                        exit()    
                    self.keyps.append(y)
                    self.keyp_ind += 1

        self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
        self.pp = torch.nn.utils.rnn.pad_sequence(self.pp, batch_first = True)
        self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)
        
        if test_state:
            self.aud = self.aud.reshape((1,-1,self.aud.shape[-1]))
            self.keyps = self.keyps.reshape((1,-1,75*3))#元々(1,-1,186)
            # self.keyps = self.keyps.reshape((1,-1,62*3))#元々(1,-1,186)
            self.pp = self.pp.reshape((1,-1,len(self.pp_ind)))
            self.data_num = 1
            self.seq = [self.aud.size(1)]

        self.aud_std, self.aud_mean = torch.std_mean(self.aud)
        self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        x = self.aud[idx]
        x2 = self.pp[idx]
        y = self.keyps[idx]
        seq_length = self.seq[idx]

        if self.state_for_test:
            self.aud_mean = self.state_for_test['aud_mean']
            self.aud_std = self.state_for_test['aud_std']
            self.keyps_mean = self.state_for_test['keyps_mean']
            self.keyps_std = self.state_for_test['keyps_std']
        
        if self.state_for_val:
            self.aud_mean = self.state_for_val['aud_mean']
            self.aud_std = self.state_for_val['aud_std']
            self.keyps_mean = self.state_for_val['keyps_mean']
            self.keyps_std = self.state_for_val['keyps_std']

        if self.transform:
            x = self.transform(x, self.aud_mean, self.aud_std)
            y = self.transform(y, self.keyps_mean, self.keyps_std) 

        return x, x2, y, seq_length
    
    
class TGM2BDataset(torch.utils.data.Dataset):
    def __init__(self, audio, keyps, 
        pp = ['string','fing','pos','ud'], transform=None, val_state=None, test_state=None):
        self.transform = transform
        self.state_for_test = test_state
        self.state_for_val = val_state
        self.aud = []
        self.pp = []
        self.keyps = []
        self.seq = []
        self.data_num = 0
        self.keyp_ind = 0

        self.aud_mean = 0
        self.aud_std = 0
        self.keyps_mean = 0
        self.keyps_std = 0
        self.pp_ind = []
        f_size = 128#128
        pp_ind_string = [0,1,2,3,4]
        pp_ind_fing = [5,6,7,8,9,10]
        pp_ind_pos = [11+i for i in range(13)]
        pp_ind_ud = [24, 25, 26]

        for p in pp:
            ns = locals()
            self.pp_ind += ns['pp_ind_' + p] 
        
        for j in range(audio.shape[1]): 
            aud_all = audio[:,j]
            
            for i in range(len(aud_all)//f_size + 1):#元々128
                x = aud_all[f_size*i:min(len(aud_all), f_size*(i+1))]
                x = torch.from_numpy(x.astype(np.float32)).clone()
                self.aud.append(x)
                self.seq.append(x.size()[0])
                self.data_num += 1
            
        for j in range(keyps.shape[1]):
            keyps_all = keyps[:,j]
            
            for i in range(len(keyps_all)//f_size + 1):
                y = keyps_all[f_size*i:min(len(keyps_all),f_size*(i+1))]
                y = torch.from_numpy(y.astype(np.float32)).clone()
                y = y.view((y.size(0),-1))
                if y.size()[0] != self.aud[self.keyp_ind].size()[0]:
                    print('Audio features and keypoint features have different lengths.')
                    print(self.aud[i].size()[0], y.size())
                    exit()    
                self.keyps.append(y)
                self.keyp_ind += 1

        self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
        # self.pp = torch.nn.utils.rnn.pad_sequence(self.pp, batch_first = True)
        self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)
        
        if test_state:
            self.aud = self.aud.reshape((1,-1,self.aud.shape[-1]))
            self.keyps = self.keyps.reshape((1,-1,15*3))#元々(1,-1,186)
            # self.keyps = self.keyps.reshape((1,-1,62*3))#元々(1,-1,186)
            # self.pp = self.pp.reshape((1,-1,len(self.pp_ind)))
            self.data_num = 1
            self.seq = [self.aud.size(1)]

        self.aud_std, self.aud_mean = torch.std_mean(self.aud)
        self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        x = self.aud[idx]
        # x2 = self.pp[idx]
        y = self.keyps[idx]
        seq_length = self.seq[idx]

        if self.state_for_test:
            self.aud_mean = self.state_for_test['aud_mean']
            self.aud_std = self.state_for_test['aud_std']
            self.keyps_mean = self.state_for_test['keyps_mean']
            self.keyps_std = self.state_for_test['keyps_std']
        
        if self.state_for_val:
            self.aud_mean = self.state_for_val['aud_mean']
            self.aud_std = self.state_for_val['aud_std']
            self.keyps_mean = self.state_for_val['keyps_mean']
            self.keyps_std = self.state_for_val['keyps_std']

        if self.transform:
            x = self.transform(x, self.aud_mean, self.aud_std)
            y = self.transform(y, self.keyps_mean, self.keyps_std) 

        return x,y, seq_length             

class woPPDataset(torch.utils.data.Dataset):
    def __init__(self, audfiles, keypsfiles, transform=None, val_state=None, test_state=None):
        self.transform = transform
        self.state_for_test = test_state
        self.state_for_val = val_state
        self.aud = []
        self.keyps = []
        self.seq = []
        self.data_num = 0
        self.keyp_ind = 0
        self.aud_mean = 0
        self.aud_std = 0
        self.keyps_mean = 0
        self.keyps_std = 0
        self.pp_ind = []
        f_size = 128#128
        # load audio features
        for f in audfiles:
            if f:
                print(f'data.py KeypsPPDataset : f in audfiles is True')
                # import IPython;IPython.embed();exit()
                aud_all = joblib.load(f)
                aud_all = aud_all.transpose(1, 0)
                for i in range(len(aud_all)//f_size + 1):#元々128
                    x = aud_all[f_size*i:min(len(aud_all), f_size*(i+1))]
                    x = torch.from_numpy(x.astype(np.float32)).clone()
                    self.aud.append(x)
                    self.seq.append(x.size()[0])
                    self.data_num += 1
            else:
                print(f'data.py KeypsPPDataset : f in audfiles is None')
        for f in keypsfiles:
            if f:
                keyps_all = joblib.load(f)
                for i in range(len(keyps_all)//f_size + 1):
                    y = keyps_all[f_size*i:min(len(keyps_all),f_size*(i+1))]
                    y = torch.from_numpy(y.astype(np.float32)).clone()
                    y = y.view((y.size(0),-1))
                    if y.size()[0] != self.aud[self.keyp_ind].size()[0]:
                        import IPython;IPython.embed();exit()
                        print('Audio features and keypoint features have different lengths.')
                        print(self.aud[i].size()[0], y.size())
                        exit()    
                    self.keyps.append(y)
                    self.keyp_ind += 1

        self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
        self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)
        
        if test_state:
            self.aud = self.aud.reshape((1,-1,self.aud.shape[-1]))
            self.keyps = self.keyps.reshape((1,-1,75*3))#元々(1,-1,186)
            self.data_num = 1
            self.seq = [self.aud.size(1)]

        self.aud_std, self.aud_mean = torch.std_mean(self.aud)
        self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        x = self.aud[idx]
        y = self.keyps[idx]
        seq_length = self.seq[idx]

        if self.state_for_test:
            self.aud_mean = self.state_for_test['aud_mean']
            self.aud_std = self.state_for_test['aud_std']
            self.keyps_mean = self.state_for_test['keyps_mean']
            self.keyps_std = self.state_for_test['keyps_std']
        
        if self.state_for_val:
            self.aud_mean = self.state_for_val['aud_mean']
            self.aud_std = self.state_for_val['aud_std']
            self.keyps_mean = self.state_for_val['keyps_mean']
            self.keyps_std = self.state_for_val['keyps_std']

        if self.transform:
            x = self.transform(x, self.aud_mean, self.aud_std)
            y = self.transform(y, self.keyps_mean, self.keyps_std) 

        return x, y, seq_length            

class woPPkeypsDataset(torch.utils.data.Dataset):
    def __init__(self, audfiles, transform=None, val_state=None, test_state=None):
        self.transform = transform
        self.state_for_test = test_state
        self.state_for_val = val_state
        self.aud = []
        self.keyps = []
        self.seq = []
        self.data_num = 0
        self.keyp_ind = 0
        self.aud_mean = 0
        self.aud_std = 0
        self.keyps_mean = 0
        self.keyps_std = 0
        self.pp_ind = []
        f_size = 128#128
        # load audio features
        for f in audfiles:
            if f:
                print(f'data.py KeypsPPDataset : f in audfiles is True')
                # import IPython;IPython.embed();exit()
                aud_all = joblib.load(f)
                aud_all = aud_all.transpose(1, 0)
                for i in range(len(aud_all)//f_size + 1):#元々128
                    x = aud_all[f_size*i:min(len(aud_all), f_size*(i+1))]
                    x = torch.from_numpy(x.astype(np.float32)).clone()
                    x_0 = torch.zeros([128,75*3])
                    self.aud.append(x)
                    self.keyps.append(x_0)
                    self.seq.append(x.size()[0])
                    self.data_num += 1
            else:
                print(f'data.py KeypsPPDataset : f in audfiles is None')
                
        self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
        self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)
        
        if test_state:
            self.aud = self.aud.reshape((1,-1,self.aud.shape[-1]))
            self.keyps = self.keyps.reshape((1,-1,75*3))#元々(1,-1,186)
            self.data_num = 1
            self.seq = [self.aud.size(1)]

        self.aud_std, self.aud_mean = torch.std_mean(self.aud)
        self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        x = self.aud[idx]
        y = self.keyps[idx]
        seq_length = self.seq[idx]
        
        if self.state_for_test:
            self.aud_mean = self.state_for_test['aud_mean']
            self.aud_std = self.state_for_test['aud_std']
            self.keyps_mean = self.state_for_test['keyps_mean']
            self.keyps_std = self.state_for_test['keyps_std']
        
        if self.state_for_val:
            self.aud_mean = self.state_for_val['aud_mean']
            self.aud_std = self.state_for_val['aud_std']
            self.keyps_mean = self.state_for_val['keyps_mean']
            self.keyps_std = self.state_for_val['keyps_std']

        if self.transform:
            x = self.transform(x, self.aud_mean, self.aud_std)
            y = self.transform(y, self.keyps_mean, self.keyps_std) 

        return x, y, seq_length            


# class KeypsDataset(torch.utils.data.Dataset):
#     def __init__(self, audfiles, keypsfiles, transform=None, test_state=None):
#         self.transform = transform
#         self.state_for_test = test_state
#         self.aud = []
#         self.keyps = []
#         self.seq = []
#         self.data_num = 0
#         self.keyp_ind = 0

#         self.aud_mean = 0
#         self.aud_std = 0
#         self.keyps_mean = 0
#         self.keyps_std = 0
        
#         # load audio features    
#         for f in audfiles:
#             if f:
#                 aud_all = joblib.load(f)
#                 aud_all = aud_all.transpose(1, 0)

#                 for i in range(len(aud_all)//128 + 1):
#                     x = aud_all[128*i:min(len(aud_all), 128*(i+1))]
#                     x = torch.from_numpy(x.astype(np.float32)).clone()
#                     self.aud.append(x)
#                     self.seq.append(x.size()[0])
#                     self.data_num += 1
            
#         # load keyps featues
#         for f in keypsfiles:
#             if f:
#                 keyps_all = joblib.load(f)
            
#                 for i in range(len(keyps_all)//128 + 1):
#                     y = keyps_all[128*i:min(len(keyps_all),128*(i+1))]
#                     y = torch.from_numpy(y.astype(np.float32)).clone()
#                     y = y.view((y.size(0),-1))
#                     if y.size()[0] != self.aud[self.keyp_ind].size()[0]:
#                         print('Audio features and keypoint features have different lengths.')
#                         print(self.aud[i].size()[0], y.size())
#                         exit()    
#                     self.keyps.append(y)
#                     self.keyp_ind += 1

#         self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
#         self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)

#         if test_state:
#             self.aud = self.aud.reshape((1,-1,128))
#             self.keyps = self.keyps.reshape((1,-1,186))
#             self.data_num = 1
#             self.seq = [self.aud.size(1)]

#         self.aud_std, self.aud_mean = torch.std_mean(self.aud)
#         self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
#     def __len__(self):
#         return self.data_num

#     def __getitem__(self, idx):
#         x = self.aud[idx]
#         y = self.keyps[idx]
#         seq_length = self.seq[idx]

#         if self.state_for_test:
#             self.aud_mean = self.state_for_test['aud_mean']
#             self.aud_std = self.state_for_test['aud_std']
#             self.keyps_mean = self.state_for_test['keyps_mean']
#             self.keyps_std = self.state_for_test['keyps_std']

#         if self.transform:
#             x = self.transform(x, self.aud_mean, self.aud_std)
#             #y = self.transform(y, self.keyps_mean, self.keyps_std) # normalization is done in the preprocess stage.

#         return x, y, seq_length             


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, audfiles, keypsfiles, transform=None, test_state=None):
#         self.transform = transform
#         self.state_for_test = test_state
#         self.aud = []
#         self.keyps = []
#         self.seq = []
#         self.data_num = 0
#         self.keyp_ind = 0

#         self.aud_mean = 0
#         self.aud_std = 0
#         self.keyps_mean = 0
#         self.keyps_std = 0
        
#         # load audio features    
#         for f in audfiles:
#             if f:
#                 aud_all = joblib.load(f)
#                 for i in range(len(aud_all)//90 + 1):
#                     x = aud_all[90*i:min(len(aud_all), 90*(i+1))]
#                     x = torch.from_numpy(x.astype(np.float32)).clone()
#                     self.aud.append(x)
#                     self.seq.append(x.size()[0])
#                     self.data_num += 1
            
#         # load keyps featues
#         for f in keypsfiles:
#             if f:
#                 keyps_all = joblib.load(f)
#                 for i in range(len(keyps_all)//90 + 1):
#                     y = keyps_all[90*i:min(len(keyps_all),90*(i+1))]
#                     y = torch.from_numpy(y.astype(np.float32)).clone()
#                     y = y[:,[0,13,12,1,2,3,4,5,6,7,8,9,10,11],:]
#                     y = y.view((y.size(0),-1))
#                     if y.size()[0] != self.aud[self.keyp_ind].size()[0]:
#                         print('Audio features and keypoint features have different lengths.')
#                         print(self.aud[i].size()[0], y.size())
#                         exit()    
#                     self.keyps.append(y)
#                     self.keyp_ind += 1

#         self.aud = torch.nn.utils.rnn.pad_sequence(self.aud, batch_first = True)
#         self.keyps = torch.nn.utils.rnn.pad_sequence(self.keyps, batch_first = True)

#         if test_state:
#             self.aud = self.aud.reshape((1,-1,28))
#             self.keyps = self.keyps.reshape((1,-1,42))
#             self.data_num = 1
#             self.seq = [self.aud.size(1)]

#         self.aud_std, self.aud_mean = torch.std_mean(self.aud)
#         self.keyps_std, self.keyps_mean = torch.std_mean(self.keyps)

    
#     def __len__(self):
#         return self.data_num

#     def __getitem__(self, idx):
#         x = self.aud[idx]
#         y = self.keyps[idx]
#         seq_length = self.seq[idx]

#         if self.state_for_test:
#             self.aud_mean = self.state_for_test['aud_mean']
#             self.aud_std = self.state_for_test['aud_std']
#             self.keyps_mean = self.state_for_test['keyps_mean']
#             self.keyps_std = self.state_for_test['keyps_std']

#         if self.transform:
#             x = self.transform(x, self.aud_mean, self.aud_std)
#             y = self.transform(y, self.keyps_mean, self.keyps_std)

#         return x, y, seq_length    

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, x, mean, std):
        x = ( x - mean ) / (std + 1e-8)
        return x
            
if __name__ == '__main__':
    # audfiles = ['../../preprocessing/audio_data/audio_features/001.jb']
    # keypsfiles = ['../../preprocessing/motion_data/coordinates_3d_aligned/001.jb']
    # dataset = KeypsDataset(audfiles, keypsfiles)
    dataset = KeypsPPDataset(0, 0, 0)
    # print(dataset.seq[-2], dataset.seq[-1])
