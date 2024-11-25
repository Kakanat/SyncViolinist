from unicodedata import bidirectional
import torch 
import torch.nn as nn
from module import  Linear
from attention import FFN_linear
from layers import Unet_block
import yaml
import torch.nn.init as init
import torch
import torch.nn as nn
# from PyxLSTM.xLSTM.block import xLSTMBlock
from itertools import chain
def oh(pred: torch.tensor):
    cls = pred.shape[1]
    pred = torch.argmax(pred,dim=1)
    onehot = torch.zeros([pred.shape[0],pred.shape[1],cls]).to(pred.device)
    for n in range(pred.shape[0]):
        for i in range(pred.shape[1]):
            onehot[n,i,pred[n,i]] = 1
    return onehot
class FBNet(nn.Module):
    """
    call LSTMLinear
    """
    def __init__(self, config):
        super(FBNet, self).__init__()

        if config['model'] == 'LSTMLinear':
            self.net = LSTMLinear(config)
        
        else:
            raise Exception('Wrong model name!')

    def forward(self, x, lengths):
        y = self.net(x, lengths)
        return y


class LSTMLinear(nn.Module):
    """
    here is the LSTM + Linear module
    all hyper parameters are initialized by config
    the weights are initialized from normal distribution mean=0.0, std=0.02
    the biases are constantly initialized as zeros

    LSTM⇒4Dense
    """
    def __init__(self, config):
        super(LSTMLinear, self).__init__()
        
        input_dim = config['input_dim']
        
        self.bidirectional = bool(config['bidirectional'])
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        # string estimation
        self.fc_s = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 5),
                                nn.LogSoftmax(dim = 2))
        # finger estimation
        self.fc_f = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 6),
                                nn.LogSoftmax(dim = 2))
        # position estimation
        self.fc_p = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 13),
                                nn.LogSoftmax(dim = 2))
        # bowing estimation
        self.fc_u = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 3),
                                nn.LogSoftmax(dim = 2))
    
        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]

        # padding process
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        
        self.lstm.flatten_parameters()
        
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4

## --motion generation --

class Model(nn.Module):
    """
    model loading according to model_type,params
    """
    def __init__(self, model_type, params):
        # super(Model, self).__init__()
        super().__init__()
        self.model_name = model_type
        # with pp model 
        if model_type == 'fc1-lstm': 
            self.net = LinearLSTM(params)
        
        elif model_type == 'fc1-lstm_wencoder': 
            self.net = LinearLSTM_wenc(params)

        elif model_type == 'fc1-lstm2': 
            self.net = LinearLSTM2(params)

        # baseline models
        elif model_type == 'A2BD':
            self.net = AudioToKeypointRNN(params)
        
        elif model_type == 'TGM2B':
            self.net = MovementNet(params)
        
        # --proposed by hiroki--
        elif model_type == 'fc1-TF':
            self.net = TransformerEncoder(params)
        
        elif model_type == 'TFe-TFd':
            self.net = TransformerEncoderDecoder(params)
        
        elif model_type == 'fc1-xlstm':
            self.net = LinearxLSTM(params)
            
        elif model_type == 'multi-lstm':
            self.net = MultiLayerLinearLSTM(params)
        elif model_type == 'TKCRNN':
            self.net = TKCRNN(params)
            
        else:
            print('Wrong name!')
            print(model_type)
            exit()
    def name(self,):
        return self.model_name
    def forward(self, x1, x2=None, lengths=None, mask = None,tgt=None, mode='train', tgt_mask = None, memory_mask = None ):
        # raise Exception(x1.shape,x2.shape,lengths)
        if 'TFd' in self.model_name:
            y = self.net(x1, x2, lengths,tgt, mode, tgt_mask, memory_mask)
        elif 'im_div' in self.model_name:
            y = self.net(x1,x2,lengths,mask)
        elif 'enc' in self.model_name:
            y = self.net(x1,lengths)
        else:
            y = self.net(x1,x2,lengths)
        return y

# --- Proposed by hiroki---
def generate_square_subsequent_mask(fs):
    mask = (torch.triu(torch.ones(fs, fs)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



class MultiLayerLinearLSTM(nn.Module):
    """
    
    """
    def __init__(self, params):
    #input_dim_aud=128, input_dim_pp=23, 
    #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
    #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        # super(LinearLSTM, self).__init__()
        super().__init__()

        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.bidirectional = params['bidirectional']
        #self.bidirectional = False
        hidden_dim = params['hidden_dim']
        # num_layers = params['num_layers']
        dropout = params['dropout']



        self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        


        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )
        
        h_init1 = nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        c_init1 = nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        self.h_init1 = nn.Parameter(h_init1, requires_grad = True)
        self.c_init1 = nn.Parameter(c_init1, requires_grad = True)
        
        h_init2 = nn.init.constant_(torch.empty(2, 1, hidden_dim), 0.0)
        c_init2 = nn.init.constant_(torch.empty(2, 1, hidden_dim), 0.0)
        self.h_init2 = nn.Parameter(h_init2, requires_grad = True)
        self.c_init2 = nn.Parameter(c_init2, requires_grad = True)
        
        h_init3 = nn.init.constant_(torch.empty(3, 1, hidden_dim), 0.0)
        c_init3 = nn.init.constant_(torch.empty(3, 1, hidden_dim), 0.0)
        self.h_init3 = nn.Parameter(h_init3, requires_grad = True)
        self.c_init3 = nn.Parameter(c_init3, requires_grad = True)

        self.lstm1 = nn.LSTM(self.input_dim_lstm, hidden_dim, 1, batch_first = True, bidirectional=self.bidirectional)
        self.lstm2 = nn.LSTM(self.input_dim_lstm, hidden_dim, 2, batch_first = True, bidirectional=self.bidirectional)
        self.lstm3 = nn.LSTM(self.input_dim_lstm, hidden_dim, 3, batch_first = True, bidirectional=self.bidirectional)
        
        self.dropout = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm2d()
        
        self.fc1 = nn.Sequential(
                        Linear(hidden_dim*(1+int(self.bidirectional)), hidden_dim*(1+int(self.bidirectional))),
                        nn.LeakyReLU(),
                        Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim),
                        nn.LeakyReLU()
                        )
        self.fc2 = nn.Sequential(
                        Linear(hidden_dim*(1+int(self.bidirectional)), hidden_dim*(1+int(self.bidirectional))),
                        nn.LeakyReLU(),
                        Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim),
                        nn.LeakyReLU()  
                        )
        self.fc3 = nn.Sequential(
                        Linear(hidden_dim*(1+int(self.bidirectional)), hidden_dim*(1+int(self.bidirectional))),
                        nn.LeakyReLU(),
                        Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim),
                        nn.LeakyReLU()
                        )
        self.fc = Linear(self.output_dim*3, self.output_dim)
        self.motion_gate1 = torch.nn.parameter.Parameter(torch.empty([1,self.output_dim]), requires_grad=True)#added        
        self.motion_gate2 = torch.nn.parameter.Parameter(torch.empty([1,self.output_dim]), requires_grad=True)#added
        self.motion_gate3 = torch.nn.parameter.Parameter(torch.empty([1,self.output_dim]), requires_grad=True)#added

        # self.fc = nn.Sequential(Linear(self.output_dim*3, self.output_dim),
        #                         nn.LeakyReLU(),
        #                         Linear(self.output_dim,self.output_dim)
        #                         )
        
        self.initialize()
    
    
    def initialize(self):
        # initialize LSTM weights and biases
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    nn.init.zeros_(param.data)
                elif 'bias_hh' in name:
                    nn.init.zeros_(param.data)
                    
        for fc in [self.fc1, self.fc2, self.fc3, self.fc]:
            for name, param in fc.named_parameters():
                if 'weight' in name:
                    if param.dim() >= 2:
                        init.xavier_uniform_(param)
                    else:
                        param.to('cuda')
                elif 'bias' in name:
                    init.constant_(param, 0)
                else:
                    param.to('cuda')

        for motion_gate in [self.motion_gate1, self.motion_gate2, self.motion_gate3]:
            init.xavier_uniform_(motion_gate.data)
            motion_gate.to('cuda')


    def forward(self, inputs_aud, inputs_pp, lengths,mask):
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        # import IPython;IPython.embed();exit()
        
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

        inputs = inputs.view(-1, total_length, self.input_dim_lstm)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        
        
        output1, (h_n, c_n) = self.lstm1(inputs, (self.h_init1.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init1.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        output1, _ = torch.nn.utils.rnn.pad_packed_sequence(output1, batch_first = True, total_length=total_length)
        
        output2, (h_n, c_n) = self.lstm2(inputs, (self.h_init2.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init2.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        output2, _ = torch.nn.utils.rnn.pad_packed_sequence(output2, batch_first = True, total_length=total_length)
        
        output3, (h_n, c_n) = self.lstm3(inputs, (self.h_init3.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init3.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        output3, _ = torch.nn.utils.rnn.pad_packed_sequence(output3, batch_first = True, total_length=total_length)
        
        output1 = output1.contiguous()
        output2 = output2.contiguous()
        output3 = output3.contiguous()
        
        output1 = output1.view(-1, output1.size()[-1])
        output2 = output2.view(-1, output2.size()[-1])
        output3 = output3.view(-1, output3.size()[-1])
        # import IPython; IPython.embed(); exit()
        output1 = self.dropout(output1)
        output2 = self.dropout(output2)
        output3 = self.dropout(output3)
        
        # output1 = self.motion_gate1*self.fc1(output1)
        # output2 = self.motion_gate2*self.fc2(output2)
        # output3 = self.motion_gate3*self.fc3(output3)
        
        # output = output1 + output2 + output3
        # import IPython;IPython.embed();exit()
        if mask==None:
            output1 = self.fc1(output1)
            output2 = self.fc2(output2)
            output3 = self.fc3(output3)
        else:
            # import IPython;IPython.embed();exit()
            output1 = self.fc1(output1) * mask[0].repeat(batch_size*total_length,1)
            output2 = self.fc2(output2) * mask[1].repeat(batch_size*total_length,1)
            output3 = self.fc3(output3) * mask[2].repeat(batch_size*total_length,1)
        
        output = self.fc(torch.cat([output1,output2,output3],dim = -1))
        output = output.view(-1, total_length, self.output_dim)
        
        previous = torch.cat([output1,output2,output3],dim = -1).view(3,-1, total_length, self.output_dim)
        
        return output,previous

class LinearLSTM(nn.Module):
    '''
    Linear Embedder + LSTM motion generator 
    weights are initialized from normal distribution mean=0.0, std=0.02 <- why 0.02?
    biases are constantly initialized as zeros
    Linear Embedder is for audio and pp?
    Emb:Linear->LeakyReLU
    cat(aud,pp)->padding->flatten->lstm(mono or bi)->padding->dropout->fc
        '''
    def __init__(self, params):
    #input_dim_aud=128, input_dim_pp=23, 
    #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
    #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        # super(LinearLSTM, self).__init__()
        super().__init__()

        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.bidirectional = params['bidirectional']
        #self.bidirectional = False
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        dropout = params['dropout']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)

        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )

        self.lstm = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm2d()
        self.fc = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)
        # import IPython;IPython.embed();exit()
        self.initialize()
    
    
    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs_aud, inputs_pp, lengths):
        # import IPython;IPython.embed();exit()
        
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        # import IPython;IPython.embed();exit()
        
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

        inputs = inputs.view(-1, total_length, self.input_dim_lstm)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm.flatten_parameters()
        
        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        # import IPython; IPython.embed(); exit()
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output
# class LinearxLSTM(nn.Module):
#     '''
#     Linear Embedder + LSTM motion generator 
#     weights are initialized from normal distribution mean=0.0, std=0.02 <- why 0.02?
#     biases are constantly initialized as zeros
#     Linear Embedder is for audio and pp?
#     Emb:Linear->LeakyReLU
#     cat(aud,pp)->padding->flatten->lstm(mono or bi)->padding->dropout->fc
#         '''
#     def __init__(self, params):
#     #input_dim_aud=128, input_dim_pp=23, 
#     #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
#     #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
#         # super(LinearLSTM, self).__init__()
#         super().__init__()

#         input_dim_aud = params['input_dim_aud']
#         input_dim_pp = params['pp_dim']
#         self.output_dim = params['output_dim']

#         self.e_dim_aud = params['e_dim_aud']
#         self.e_dim_pp = params['e_dim_pp']

#         self.bidirectional = params['bidirectional']
#         #self.bidirectional = False
#         hidden_dim = params['hidden_dim']
#         num_layers = params['num_layers']
#         dropout = params['dropout']
#         self.num_blocks = 1#xLSTMのブロック数
#         h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
#         c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

#         self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        
#         self.h_init = nn.Parameter(h_init, requires_grad = True)
#         self.c_init = nn.Parameter(c_init, requires_grad = True)
#         self.lstm_type = 'slstm'
#         self.aud_emb = nn.Sequential(
#                         Linear(input_dim_aud, self.e_dim_aud),
#                         nn.LeakyReLU()
#                         )
        
#         self.pp_emb = nn.Sequential(
#                         Linear(input_dim_pp, self.e_dim_pp),
#                         nn.LeakyReLU()
#                         )

#         # self.lstm = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
#         self.blocks = nn.ModuleList([xLSTMBlock(self.input_dim_lstm if i == 0 else hidden_dim,
#                                                 hidden_dim,self.output_dim, num_layers, dropout, self.bidirectional, 'slstm')
#                                     for i in range(self.num_blocks)])
#         self.dropout = nn.Dropout(dropout)
#         # self.bn2 = nn.BatchNorm2d()
#         self.fc = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)

#         # self.initialize()
    
    
#     def initialize(self):
#         # initialize LSTM weights and biases
#         for layer in self.block._all_weights:
#             for param_name in layer:
#                 if 'weight' in param_name:
#                     weight = getattr(self.block, param_name)
#                     torch.nn.init.normal_(weight.data, 0.0, 0.02)
#                 else:
#                     bias = getattr(self.block, param_name)
#                     nn.init.constant_(bias.data, 0.0)

#     def forward(self, inputs_aud, inputs_pp, lengths, hidden_states=None):
#         batch_size = inputs_aud.size()[0]
#         total_length = inputs_aud.size()[1]
        
#         inputs_aud = inputs_aud.contiguous()
#         inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
#         inputs_pp = inputs_pp.contiguous()
#         inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

#         inputs_aud = self.aud_emb(inputs_aud)
#         inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
#         inputs_pp = self.pp_emb(inputs_pp)
#         inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
#         # import IPython;IPython.embed();exit()
        
#         inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

#         inputs = inputs.view(-1, total_length, self.input_dim_lstm)
#         # inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
#         # self.lstm.flatten_parameters()
        
#         if hidden_states is None:
#             hidden_states = [None] * self.num_blocks
#         for i, block in enumerate(self.blocks):
#             output, hidden_state = block(inputs, hidden_states[i])
#             if self.lstm_type == "slstm":
#                 hidden_states[i] = [[hidden_state[j][0], hidden_state[j][1]] for j in range(len(hidden_state))]
#             assert not torch.isnan(output).any(), "output has NaN"
        
#         # output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
#         #                                 self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
#         # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)

#         output = output.contiguous()
#         # import IPython; IPython.embed(); exit()
        
#         output = output.view(-1, output.size()[-1])
#         # output = self.dropout(output)
        
#         # output = self.fc(output)
#         output = output.view(-1, total_length, self.output_dim)
#         return output


class TransformerEncoderDecoder(nn.Module):
    """
    Linear Embedder + Transformer motion generator
    """
    
    def __init__(self,params):
        super().__init__()
        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        # dmodel_e = params['dmodel_e']
        # dmodel_d = params['dmodel_d']
        num_layers_e = params['num_layers_e']
        num_layers_d = params['num_layers_d']
        num_head_e = params['num_head_e']
        num_head_d = params['num_head_d']
        num_fc = params['num_fc']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.input_dim = self.e_dim_aud + self.e_dim_pp
        
        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=num_head_e, dim_feedforward=self.input_dim, dropout=0.1, activation=torch.nn.LayerNorm(self.input_dim), layer_norm_eps=1e-05, batch_first=True, norm_first=False)
        self.transformer_e = torch.nn.TransformerEncoder(self.encoder_layer, num_layers_e, norm=torch.nn.LayerNorm(self.input_dim))
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model = self.output_dim, nhead = num_head_d, dim_feedforward=self.output_dim, dropout=0.1, activation=torch.nn.LayerNorm(self.output_dim), layer_norm_eps=1e-05, batch_first=True, norm_first=False)
        self.transformer_d = torch.nn.TransformerDecoder(self.decoder_layer, num_layers_d, norm=torch.nn.LayerNorm(self.output_dim))
        self.relu = torch.nn.LeakyReLU()
        self.encoder_fc = nn.Sequential(
            self.relu,
            Linear(self.input_dim,self.output_dim)
            )
        fc = []
        for _ in range(num_fc-1):
            fc.extend([self.relu,Linear(self.input_dim,self.input_dim)])
        fc.extend([nn.LeakyReLU(),Linear(self.input_dim,self.output_dim)])
        self.decoder_fc = nn.Sequential(*fc)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.transformer_e.named_parameters() + self.transformer_d.named_parameters() + self.encoder_fc.named_parameters() + self.decoder_fc.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if param.dim() >= 2:
                    init.xavier_uniform_(param)
                else:
                    param.to('cuda')
            elif 'bias' in name and 'norm' not in name:
                init.constant_(param, 0)
            else:
                param.to('cuda')
                
                
    def name():
        return 'LinearTransformer'
        
    def forward(self,inputs_aud, inputs_pp,lengths, tgt=None, mode='train', tgt_mask = None, memory_mask = None):
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)
        # import IPython;IPython.embed();exit()
        memory = self.transformer_e(inputs)
        memory = self.encoder_fc(memory)
        
        if mode == 'train':                
            output = self.transformer_d(tgt, memory,tgt_mask,memory_mask)
        else:
            tgt_input = torch.zeros(batch_size, 1, memory.size(2)).to(inputs.device)
            for _ in range(memory.size(1)):
                out = self.transformer_d(tgt_input, memory, memory_mask=memory_mask)
                tgt_input = torch.cat([tgt_input, out], dim=1)
            output = torch.cat(output, dim=1)
        return output



class TransformerEncoder(nn.Module):
    """
    Linear Embedder + Transformer encoder motion generator
    """
    
    def __init__(self,params):
        super().__init__()
        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        # dmodel = params['mtoken']
        num_elayers = params['num_elayers']
        num_head = params['num_head']
        num_fc = params['num_fc']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.input_dim = self.e_dim_aud + self.e_dim_pp
        
        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=num_head)
        self.transformer_e = torch.nn.TransformerEncoder(self.encoder_layer, num_elayers, norm=torch.nn.LayerNorm(self.input_dim), enable_nested_tensor=True, mask_check=True)
        
        self.relu = torch.nn.LeakyReLU()
        fc = []
        for i in range(num_fc-1):
            fc.extend([nn.LeakyReLU(),Linear(self.input_dim,self.input_dim)])
        fc.extend([nn.LeakyReLU(),Linear(self.input_dim,self.output_dim)])
        self.fc = nn.Sequential(*fc)
        self._init_weights()

    def _init_weights(self):
        # Initialize transformer parameters
        for name, param in chain(self.transformer_e.named_parameters(), self.fc.named_parameters()):
            if 'weight' in name and 'norm' not in name:
                if param.dim() >= 2:
                    init.xavier_uniform_(param)
                else:
                    param.to('cuda')
            elif 'bias' in name and 'norm' not in name:
                init.constant_(param, 0)
            else:
                param.to('cuda')
    def name():
        return 'LinearTransformer'
        
    def forward(self,inputs_aud, inputs_pp, lengths):
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

        # inputs = inputs.view(-1, self.input_dim, total_length)
        # import IPython;IPython.embed();exit()
        output = self.transformer_e(inputs)
        output = self.relu(output)
        # import IPython;IPython.embed();exit()
        # output = output.view(batch_size,total_length,self.input_dim)
        output = self.fc(output)
        # output = output.view(batch_size,total_length,self.output_dim)
        return output

class LinearLSTM2(nn.Module):
    '''
    Linear Embedder + LSTM motion generator 
    weights are initialized from normal distribution mean=0.0, std=0.02 <- why 0.02?
    biases are constantly initialized as zeros
    Linear Embedder is for audio and pp?
    Emb:Linear->LeakyReLU
    cat(aud,pp)->padding->flatten->lstm(mono or bi)->padding->dropout->fc
        '''
    def __init__(self, params):
    #input_dim_aud=128, input_dim_pp=23, 
    #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
    #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        # super(LinearLSTM, self).__init__()
        super().__init__()

        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.bidirectional = params['bidirectional']
        #self.bidirectional = False
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        dropout = params['dropout']

        h_init1 = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init1 = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        h_init2 = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init2 = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        
        self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        
        self.h_init1 = nn.Parameter(h_init1, requires_grad = True)
        self.c_init1 = nn.Parameter(c_init1, requires_grad = True)
        self.h_init2 = nn.Parameter(h_init2, requires_grad = True)
        self.c_init2 = nn.Parameter(c_init2, requires_grad = True)
        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )

        self.lstm1 = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)
        self.lstm2 = nn.LSTM(self.output_dim, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm1._all_weights + self.lstm2._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm1, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm1, param_name)
                    nn.init.constant_(bias.data, 0.0)


    def forward(self, inputs_aud, inputs_pp, lengths):
        lengths_out = torch.tensor([self.output_dim]*len(lengths))
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        # import IPython; IPython.embed(); exit()
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

        inputs = inputs.view(-1, total_length, self.input_dim_lstm)
        # print(inputs.shape)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        # print(inputs[0].shape,inputs[1].shape)
        output, (h_n, c_n) = self.lstm1(inputs, (self.h_init1.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init1.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        # print(output[0].shape,output[1].shape)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        # print(output.shape)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        # print(output.shape)
        output = self.dropout(output)
        output = self.fc1(output)
        # print(output.shape)

        # import IPython; IPython.embed(); exit()
        output = output.view(-1, total_length, self.output_dim)
        # print(output.shape)

        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first = True)
        # print(output[0].shape,output[1].shape)

        output, (h_n, c_n) = self.lstm2(output, (self.h_init2.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init2.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        # print(output[0].shape,output[1].shape)
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        # print(output.shape)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        output = self.dropout(output)
        output = self.fc2(output)
        output = output.view(-1, total_length, self.output_dim)
        return output
    
class LinearLSTM_wenc(nn.Module):
    '''
    Linear Embedder + LSTM motion generator 
    weights are initialized from normal distribution mean=0.0, std=0.02 <- why 0.02?
    biases are constantly initialized as zeros
    Linear Embedder is for audio and pp?
    Emb:Linear->LeakyReLU
    cat(aud,pp)->padding->flatten->lstm(mono or bi)->padding->dropout->fc
        '''
    def __init__(self, params):
    #input_dim_aud=128, input_dim_pp=23, 
    #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
    #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        # super(LinearLSTM, self).__init__()
        super().__init__()

        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.bidirectional = params['bidirectional']
        #self.bidirectional = False
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        dropout = params['dropout']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)

        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )

        self.lstm = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm2d()
        self.fc = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)
        # import IPython;IPython.embed();exit()
        self.initialize()
        
        model_pth1 = '../FBestimation/results/TKCRNN_0724/mfcc_28_ws/'
        model_pth2 = '../FBestimation/results/TKCRNN_0724/mfcc_28_wf/'
        model_pth3 = '../FBestimation/results/TKCRNN_0724/mfcc_28_wp/'
        model_pth4 = '../FBestimation/results/TKCRNN_0724/mfcc_28_wu/'

        with open(model_pth1+'config.yaml') as f:
            config1 = yaml.safe_load(f) 
        with open(model_pth2+'config.yaml') as f:
            config2 = yaml.safe_load(f) 
        with open(model_pth3+'config.yaml') as f:
            config3 = yaml.safe_load(f) 
        with open(model_pth4+'config.yaml') as f:
            config4 = yaml.safe_load(f) 
        
        self.net_f1 = FBNet(config1)
        self.net_f1 = self.net_f1
        self.net_f1.load_state_dict(torch.load(model_pth1+'FBNet_best.pth'))
        self.net_f1.eval()
        self.net_f2 = FBNet(config2)
        self.net_f2 = self.net_f2
        self.net_f2.load_state_dict(torch.load(model_pth2+'FBNet_best.pth'))
        self.net_f2.eval()
        self.net_f3 = FBNet(config3)
        self.net_f3 = self.net_f3
        self.net_f3.load_state_dict(torch.load(model_pth3+'FBNet_best.pth'))
        self.net_f3.eval()
        self.net_f4 =FBNet(config4)
        self.net_f4 = self.net_f4
        self.net_f4.load_state_dict(torch.load(model_pth4+'FBNet_best.pth'))
        self.net_f4.eval()
    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs_aud,  lengths):
        # import IPython;IPython.embed();exit()
        
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        # inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        # print(lengths)
    
        x2 = torch.cat(
            [oh(self.net_f1(inputs_aud,lengths)[0])\
            ,oh(self.net_f2(inputs_aud,lengths)[1])\
            ,oh(self.net_f3(inputs_aud,lengths)[2])\
            ,oh(self.net_f4(inputs_aud,lengths)[3])
            ],dim=2)
        
        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        x2 = x2.contiguous()
        x2 = x2.view(-1, x2.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        x2 = self.pp_emb(x2)
        x2 = x2.view(-1, total_length, self.e_dim_pp)
        
        inputs = torch.cat((inputs_aud, x2), dim=2)

        inputs = inputs.view(-1, total_length, self.input_dim_lstm)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm.flatten_parameters()

        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output
    
    
# --- Proposed by hirata---
class HirataLinearLSTM(nn.Module):
    '''
    Linear Embedder + LSTM motion generator 
    weights are initialized from normal distribution mean=0.0, std=0.02 <- why 0.02?
    biases are constantly initialized as zeros
    Linear Embedder is for audio and pp?
    Emb:Linear->LeakyReLU
    cat(aud,pp)->padding->flatten->lstm(mono or bi)->padding->dropout->fc
        '''
    def __init__(self, params):
    #input_dim_aud=128, input_dim_pp=23, 
    #            output_dim_aud=128, output_dim_pp=16, output_dim=186, 
    #            hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        # super(LinearLSTM, self).__init__()
        super().__init__()

        input_dim_aud = params['input_dim_aud']
        input_dim_pp = params['pp_dim']
        self.output_dim = params['output_dim']

        self.e_dim_aud = params['e_dim_aud']
        self.e_dim_pp = params['e_dim_pp']

        self.bidirectional = params['bidirectional']
        #self.bidirectional = False
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        dropout = params['dropout']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        self.input_dim_lstm = self.e_dim_aud + self.e_dim_pp
        
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)

        self.aud_emb = nn.Sequential(
                        Linear(input_dim_aud, self.e_dim_aud),
                        nn.LeakyReLU()
                        )
        
        self.pp_emb = nn.Sequential(
                        Linear(input_dim_pp, self.e_dim_pp),
                        nn.LeakyReLU()
                        )

        self.lstm = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm2d()
        self.fc = Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)
        # import IPython;IPython.embed();exit()
        self.initialize()
    
    
    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs_aud, inputs_pp, lengths):
        # import IPython;IPython.embed();exit()
        
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        inputs_pp = inputs_pp.contiguous()
        inputs_pp = inputs_pp.view(-1, inputs_pp.size()[-1])

        inputs_aud = self.aud_emb(inputs_aud)
        inputs_aud = inputs_aud.view(-1, total_length, self.e_dim_aud)
        inputs_pp = self.pp_emb(inputs_pp)
        inputs_pp = inputs_pp.view(-1, total_length, self.e_dim_pp)
        # import IPython;IPython.embed();exit()
        
        inputs = torch.cat((inputs_aud, inputs_pp), dim=2)

        inputs = inputs.view(-1, total_length, self.input_dim_lstm)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm.flatten_parameters()
        
        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        # import IPython; IPython.embed(); exit()
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output



class LinearLSTMwoPP(nn.Module):
    '''
    Linear Embedder + LSTM motion generator 
    Embing:Linear
    aud->emb->padding->lstm->padding->dropout->fc
    '''
    def __init__(self, input_dim_aud=128, output_dim_aud=128, output_dim=186, 
                hidden_dim=512, num_layers = 1, bidirectional=False, dropout=0.1):
        super(LinearLSTMwoPP, self).__init__()

        self.output_dim_aud = output_dim_aud
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        num_layers = num_layers
        self.input_dim_lstm = output_dim_aud
        
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)

        self.aud_emb = Linear(input_dim_aud, output_dim_aud)
        
        self.lstm = nn.LSTM(self.input_dim_lstm, hidden_dim, num_layers, batch_first = True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = Linear(hidden_dim*(1+int(self.bidirectional)), output_dim)

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs_aud, inputs_pp, lengths):
        batch_size = inputs_aud.size()[0]
        total_length = inputs_aud.size()[1]
        
        inputs_aud = inputs_aud.contiguous()
        inputs_aud = inputs_aud.view(-1, inputs_aud.size()[-1])
        
        inputs_aud = self.aud_emb(inputs_aud)
        inputs = inputs_aud.view(-1, total_length, self.output_dim_aud)
        
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm.flatten_parameters()

        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output


#---A2BD

class AudioToKeypointRNN(nn.Module):
    
    '''
    
    LSTM motion generator (based on A2BD and TGM2B)
    padding->lstm->paddin->dropout->fc
    
    '''
    
    def __init__(self, params): 
    #input_dim=128, hidden_dim=512, num_layers=1, output_dim=186, dropout=0.1, bidirectional=False):
        super(AudioToKeypointRNN, self).__init__()

        self.output_dim = params['output_dim']
        self.bidirectional = params['bidirectional']
        num_layers = params['num_layers']
        hidden_dim = params['hidden_dim']
        input_dim = params['input_dim']

        dropout = params['dropout'] * int(num_layers > 1) 

        # trainable h & c
        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim), 0.0)

        num_layers = num_layers
        
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*(1+int(self.bidirectional)), self.output_dim)

        self.initialize()


    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
        # initialize fc layer
            nn.init.xavier_normal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias.data, 0)


    def forward(self, inputs, dummy_inputs, lengths):
        batch_size = inputs.size()[0]
        total_length = inputs.size()[1]
        # import IPython;IPython.embed();exit()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                        self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        output = output.contiguous()
        output = output.view(-1, output.size()[-1])
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output



#---TGM2B---

class HandEncoder(nn.Module):
    """
    HandEncoder with Self-attention and U-net
    loading Unet in layers.py & ffn in attention.py

    """
    def __init__(self, d_input, d_model, n_block, n_unet, n_attn, n_head, max_len, dropout,
                pre_lnorm, attn_type):
        
        super(HandEncoder, self).__init__()
        self.linear = Linear(d_input, d_model)
        self.unet = nn.ModuleList([Unet_block(d_model, n_unet, n_attn, n_head, max_len, dropout, pre_lnorm, attn_type)] * n_block)
        self.ffn = FFN_linear(d_model, dropout)
        
    def forward(self, enc_input, lengths, return_attns=False):
        x = self.linear(enc_input)
        for unet in self.unet:
            x = unet(x, lengths, return_attns)      
        enc_output = self.ffn(x)
        return enc_output
        
class TGM2BGenerator(nn.Module):
    """
    LSTM Generator
    padding->flatten->lstm->padding->dropout->fc
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TGM2BGenerator, self).__init__()       
        self.output_dim = output_dim
        
        # Trainable h & c
        h_init = \
            nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        c_init = \
            nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad=True)
        self.c_init = nn.Parameter(c_init, requires_grad=True)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = Linear(hidden_dim, output_dim)        

        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(0)
        total_length = inputs.size(1)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1, batch_size, 1), self.c_init.repeat(1, batch_size, 1)))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size(-1))  # flatten before FC
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output

class MovementNet(nn.Module):
    """
    Full Body Movement Network
    body -> Generator
    hand -> Generator(hand_encoded) + refine_net(hand_encoded:for 3: landmarks)
    """
    def __init__(self, params):
    # d_input=28, d_output_body=36, d_output_rh=6, d_model=512, n_block=2, n_unet=4, n_attn=1, n_head=4,  max_len=90, dropout=0.1, 
    #              pre_layernorm=False, attn_type='rel', gpu='0'):
        super(MovementNet, self).__init__()
        self.gpu = params['gpu']
        d_input = params['d_input']
        d_output_body = params['d_output_body']
        d_output_rh = params['d_output_rh']
        d_model = params['d_model']
        n_block = params['n_block']
        n_unet = params['n_unet']
        n_attn = params['n_attn']
        n_head = params['n_head']
        max_len = params['max_len']
        pre_layernorm = params['pre_layernorm']
        attn_type = params['attn_type']
        dropout = params['dropout']

        self.bodynet = TGM2BGenerator(d_input, d_model, d_output_body, dropout)
        self.handencoder = HandEncoder(d_input, d_model, n_block, n_unet, n_attn, n_head, max_len, dropout, 
                                    pre_layernorm, attn_type)
        self.handdecoder = TGM2BGenerator(d_model, d_model, d_output_rh, dropout)
        self.refine_network = Linear(d_model, 3)
        
    def forward(self, inputs, dummy_inputs, lengths, return_attns=False):
        """
        Args: 
            inputs: [B, T, D]
            lengths: [B]  
        Returns:
            output: [B, T, (K*3)]
        """
        body_output = self.bodynet(inputs, lengths)
        enc_output = self.handencoder.forward(inputs, lengths, return_attns=return_attns)
        rh_output = self.handdecoder.forward(enc_output, lengths)
        rh_refined = self.refine_network(enc_output)
        # import IPython;IPython.embed();exit()
        rh_refined = rh_output[:, :, 3:] + rh_refined
        rh_final = torch.cat([rh_output[:, :, :3], rh_refined], dim=-1)
        full_output = torch.cat([body_output[:,:,:33], rh_final, body_output[:,:,33:]], dim=-1)
        return full_output

def L2loss(pred, target, mask):
    diff = torch.abs(pred - target)**2
    out = torch.sum(diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)
    

def L1Loss(pred, target, mask):
    diff = torch.abs(pred - target)
    out = torch.sum(diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)

# for debug
if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy_x = torch.randn(10, 128, 128).to(device)
    dummy_x2 = torch.randn(10, 128, 23).to(device)

    lengths = [128,128,128,128,128,128,128,128,128,100] 
    dummy_y = torch.randn(10, 128, 186).to(device)
    mask = dummy_y != 0
    mask = mask.to(device)
    device = 0
    
    with open('./configs/config.yaml') as f:
        params = yaml.safe_load(f)
    
    net = Model(params['model'], params['params']).to(device)
    print(net)
    #net = LinearLSTM(params['params']).to(device)
    #net = LinearLSTMwoPP().to(device)
    
    
    #net = AudioToKeypointRNN(num_layers = 1, bidirectional=False).to(device)

    #out = net(dummy_x, lengths)
    out = net(dummy_x, dummy_x2, lengths)
    print(out.size()) # 10 x 128 x 186
    exit()

    loss1 = L1Loss(out, dummy_y, mask[:,:,:1])
    criterion = nn.L1Loss(reduction='mean')
    loss2 = criterion(out, dummy_y)

    print(loss2.mean())





import torch 
import torch.nn as nn
from module import Linear, TemporalBlock, FBClassifier
#from torchinfo import summary

class FBNet(nn.Module):
    def __init__(self, config):
        super(FBNet, self).__init__()

        if config['model'] == 'LSTMLinear':
            self.net = LSTMLinear(config)

        elif config['model'] == 'CNN':
            self.net = ConvNet(config)
        
        elif config['model'] == 'TCN':
            self.net = TemporalConvNet(config)

        elif config['model'] == 'CRNN':
            self.net = CRNN(config)

        elif config['model'] == 'TKCRNN':
            self.net = TKCRNN(config)
        
        elif config['model'] == 'TKCRNN_0724':
            self.net = TKCRNN_0724(config)
            
        elif config['model'] == 'NHCRNN':
            self.net = NHCRNN0609(config)
        elif config['model'] == 'C3LSTM':
            self.net = NHCRNN0609(config)
        elif config['model'] == 'C1LSTM':
            self.net = C1LSTM(config)
        elif config['model'] == 'C2LSTM':
            self.net = C2LSTM(config)
        elif config['model'] == 'C3RNN':
            self.net = C3RNN(config)
        elif config['model'] == 'C3GRU':
            self.net = C3GRU(config)
        elif config['model'] == 'C4LSTM':
            self.net = C4LSTM(config)
        elif config['model'] == 'C5LSTM':
            self.net = C5LSTM(config)
        elif config['model'] == 'CSC5LSTM':
            self.net = CSC5LSTM(config)
        else:
            raise Exception('Wrong model name!')

    def forward(self, x, lengths=None):
        y = self.net(x, lengths)
        return y


class LSTMLinear(nn.Module):
    def __init__(self, config):
        super(LSTMLinear, self).__init__()
        
        input_dim = config['input_dim']
        
        self.bidirectional = bool(config['bidirectional'])
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 5),
                                nn.LogSoftmax(dim = 2))
        self.fc_f = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 6),
                                nn.LogSoftmax(dim = 2))
        
        self.fc_p = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 13),
                                nn.LogSoftmax(dim = 2))
    
        self.fc_u = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 3),
                                nn.LogSoftmax(dim = 2))
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4


class TKCRNN(nn.Module):
    def __init__(self, config):
        super(TKCRNN, self).__init__()
        
        input_dim = config['input_dim']

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(4,1)),
            nn.Dropout(p=0)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=0)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=0)
        )
        
        self.bidirectional = bool(config['bidirectional'])
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(input_dim*8, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 5),
                                nn.LogSoftmax(dim = 2))
        self.fc_f = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 6),
                                nn.LogSoftmax(dim = 2))
        
        self.fc_p = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 13),
                                nn.LogSoftmax(dim = 2))
    
        self.fc_u = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 3),
                                nn.LogSoftmax(dim = 2))
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]

        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = x.permute((0, 3, 1, 2)) # B x T x C x F 
        # CNNの出力をRNNの入力に変形する
        x = x.view(x.size(0), x.size(1), -1) # B x T x (C x F)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4


class ConvNet(nn.Module):
    def ___init__(self):
        pass


class TemporalConvNet(nn.Module):
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        num_inputs = config['input_dim']
        num_channels = config['tcn_hidden_dims']
        kernel_size = config['kernel_size']
        dropout = config['dropout']
        hidden_fc = config['fc_hidden_dim']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size) //2*dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                    padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.dropout3 = nn.Dropout(dropout)
        self.fb = FBClassifier(num_channels[-1], hidden_dim=hidden_fc)

    def forward(self, x, lengths=None):
        x = x.permute((0,2,1)) # -> B x C x T
        x = self.network(x) # -> B x C x T
        x = self.dropout3(x)
        x = x.permute((0,2,1))
        y1, y2, y3, y4 = self.fb(x)
        return y1, y2, y3, y4


class CRNN(nn.Module):
    def __init__(self, config):        
    #, input_size=(1), hidden_size=[128, 128, 128, 128], num_layers=2, num_classes=3, kernel_size=3, dropout=0.5):
        super(CRNN, self).__init__()

        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_fc = config['fc_hidden_dim']
        #kernel_size = config['kernel_size']
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=hidden_size[0], kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=hidden_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1)),
            nn.Dropout(p=dropout)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size[0], out_channels=hidden_size[1], kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=hidden_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=dropout)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size[1], out_channels=hidden_size[2], kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=hidden_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=dropout)
        ) 

        # RNN層を定義する
        self.rnn = nn.LSTM(input_size=hidden_size[2]*8, hidden_size=hidden_size[3], num_layers=num_layers, bidirectional=True)

        # 全結合層を定義する
        #self.fc = nn.Linear(in_features=hidden_size[3]*2, out_features=num_classes)
        self.fb = FBClassifier(hidden_size[-1]*2, hidden_dim = hidden_fc)

    def forward(self, x, length=None):
        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = x.permute((0, 3, 1, 2)) # B x T x C x F 
        # CNNの出力をRNNの入力に変形する
        x = x.view(x.size(0), x.size(1), -1) # B x T x (C x F)
        # RNN層を適用する
        x, _ = self.rnn(x) # B x T x H
        #print(x.size())
        # RNNの出力を全結合層に入力する
        y1, y2, y3, y4 = self.fb(x) # B x cls x T
        return y1, y2, y3, y4


## for debug
if __name__ == '__main__':
    ### aud: [N, T, C]
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # net = BLSTM('sf').to(device)
    # net.train()
    # dummy_x = torch.randn(10, 1876, 128).to(device)
    # dummy_y = torch.randint(4, (10, 1876)).to(device)
    # out = net(dummy_x)
    
    # loss = torch.nn.NLLLoss()

    # loss = loss(torch.log(out), dummy_y)
    # print(out.argmax(dim=1), out.size())
    # name = 'LSTMLinear'
    # config = {
    #     'model': 'LSTMLinear', 
    #     'input_dim': 128,
    #     'bidirectional': 0,
    #     'dropout': 0.1,
    #     'lstm_hidden_dim': 512,
    #     'fc_hidden_dim': 256,
    #     'num_layers': 1
    # }

    name = 'TCN'
    config = {
        'model': 'TCN', 
        'input_dim': 128,
        'tcn_hidden_dims': [64, 64, 64, 64],
        'kernel_size': 3,
        'dropout': 0.2,
        'lstm_hidden_dim': 512,
        'fc_hidden_dim': 32,
        'num_layers': 1
    }

    B = 10

    net = FBNet(config).to(device)
    net.train()
    dummy_x = torch.randn(10, 128, 128).to(device) # -> N x T x C
    #print(summary(model=net, input_size=(B, 128, 128)))
    dummy_y = torch.randint(4, (10, 128)).to(device)

    dummy_len = [128, 128, 128, 128, 128, 128, 128, 128, 128, 100]
    # y1, y2, y3, y4 = net(dummy_x, dummy_len)
    # print(y1.size(), y2.size(), y3.size(), y4.size())
    
    y = net(dummy_x, dummy_len)
    print(y[0].size()) # -> B x C x T
    
    criterion = torch.nn.NLLLoss(reduction='none')

    loss = criterion(y[0], dummy_y)
    print(loss.size())
    exit()

    dummy_y = torch.zeros_like(torch.empty(10, 128)).to(device)
    
    loss = torch.nn.BCELoss(reduction='mean')

    loss = loss(y4, dummy_y)
    print(loss, y4.size())

        
class TKCRNN_0724(nn.Module):
    def __init__(self, config):
        super(TKCRNN_0724, self).__init__()
        
        input_dim = config['input_dim']

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(4,1)),
            nn.Dropout(p=0)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=0)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(p=0)
        )
        
        self.bidirectional = bool(config['bidirectional'])
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(128, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 5),
                                nn.LogSoftmax(dim = 2))
        self.fc_f = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 6),
                                nn.LogSoftmax(dim = 2))
        
        self.fc_p = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 13),
                                nn.LogSoftmax(dim = 2))
    
        self.fc_u = nn.Sequential(
                                Linear(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2),
                                nn.LeakyReLU(),
                                Linear(hidden_dim_2, 3),
                                nn.LogSoftmax(dim = 2))
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        # x = x.reshape(len(lengths), -1, 28)
        batch_size = x.size()[0]
        total_length = x.size()[1]
        
        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = x.permute((0, 3, 1, 2)) # B x T x C x F 
        # CNNの出力をRNNの入力に変形する
        x = x.view(x.size(0), x.size(1), -1) # B x T x (C x F)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
        
class NHCRNN0609(nn.Module):
    def __init__(self, config):
        super(NHCRNN0609, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(input_dim*input_dim/(2**3))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']
        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn3 = Conv2d(64,128,(3,3),"same",nn.GELU(),(2,1),0)
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = x.permute((0,3,2,1)) # B x T × F x C
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)
        # import IPython;IPython.embed();exit()
        # RNN層を適用する
        # print(lengths)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
    
    
class C1LSTM(nn.Module):
    def __init__(self, config):
        super(C1LSTM, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(input_dim*32/(2**1))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']
        
        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),0)
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]

        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        
        x = x.permute((0,3,2,1)) # B x T × F x C 
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
    
class C2LSTM(nn.Module):
    def __init__(self, config):
        super(C2LSTM, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(input_dim*64/(2**2))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']
        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),0)
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]

        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        
        x = x.permute((0,3,2,1)) # B x T × F x C 
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
    
    
class C4LSTM(nn.Module):
    def __init__(self, config):
        super(C4LSTM, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(128*256/(2**4))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn3 = Conv2d(64,128,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn4 = Conv2d(128,256,(3,3),"same",nn.GELU(),(2,1),0)
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = self.cnn4(x)
        x = x.permute((0,3,2,1)) # B x T × F x C
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
    
    
class C3RNN(nn.Module):
    def __init__(self, config):
        super(C3RNN, self).__init__()
        
        input_dim = config['input_dim']
        self.input_rnn = int(input_dim*input_dim/(2**3))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),dropout)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),dropout)
        self.cnn3 = Conv2d(64,128,(3,3),"same",nn.GELU(),(2,1),dropout)
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        # self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.rnn = nn.RNN(self.input_rnn, hidden_dim_1, num_layers,nonlinearity='tanh', batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
    

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.rnn._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.rnn, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.rnn, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x = x.permute((0, 3, 1, 2)) # B x T x C x F 
        # CNNの出力をRNNの入力に変形する
        x = x.view(x.size(0), x.size(1), -1) # B x T x (C x F)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # import IPython;IPython.embed();exit()
        self.rnn.flatten_parameters()
        output, (h_n, c_n) = self.rnn(x, self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4
    
class C3GRU(nn.Module):
    def __init__(self, config):
        super(C3GRU, self).__init__()
        
        input_dim = config['input_dim']
        self.input_gru = int(input_dim*input_dim/(2**3))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),dropout)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),dropout)
        self.cnn3 = Conv2d(64,128,(3,3),"same",nn.GELU(),(2,1),dropout)
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        # self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.gru = nn.GRU(self.input_gru, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        
        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.gru._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.gru, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.gru, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x = x.permute((0, 3, 1, 2)) # B x T x C x F 
        # CNNの出力をRNNの入力に変形する
        x = x.view(x.size(0), x.size(1), -1) # B x T x (C x F)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        
        self.gru.flatten_parameters()
        output, (h_n, c_n) = self.gru(x, self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4

class C5LSTM(nn.Module):
    def __init__(self, config):
        super(C5LSTM, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(128*512/(2**5))
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        self.cnn1 = Conv2d(1,32,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn2 = Conv2d(32,64,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn3 = Conv2d(64,128,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn4 = Conv2d(128,256,(3,3),"same",nn.GELU(),(2,1),0)
        self.cnn5 = Conv2d(256,512,(3,3),"same",nn.GELU(),(2,1),0)
        
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = self.cnn4(x)
        x = self.cnn5(x)
        
        x = x.permute((0,3,2,1)) # B x T × F x C
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4

class CSC5LSTM(nn.Module):
    def __init__(self, config):
        super(CSC5LSTM, self).__init__()
        
        input_dim = config['input_dim']
        self.input_lstm = int(128*32)
        num_layers = config['num_layers']
        dropout = config['dropout']
        hidden_dim_1 = config['lstm_hidden_dim']
        hidden_dim_2 = config['fc_hidden_dim']

        self.cnn1 = Conv2dSC(1,32,(3,3),"same",nn.GELU(),0)
        self.cnn2 = Conv2dSC(32,32,(3,3),"same",nn.GELU(),0)
        self.cnn3 = Conv2dSC(32,32,(3,3),"same",nn.GELU(),0)
        self.cnn4 = Conv2dSC(32,32,(3,3),"same",nn.GELU(),0)
        self.cnn5 = Conv2dSC(32,32,(3,3),"same",nn.GELU(),0)
        
        
        self.bidirectional = bool(config['bidirectional'])

        h_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        c_init = nn.init.constant_(torch.empty(num_layers, 1, hidden_dim_1), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad = True)
        self.c_init = nn.Parameter(c_init, requires_grad = True)
        
        self.lstm = nn.LSTM(self.input_lstm, hidden_dim_1, num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.fc_s = String_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_f = Fingering_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_p = Position_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)
        self.fc_u = Bowing_Decoder(hidden_dim_1 * (1 + int(self.bidirectional)), hidden_dim_2)

        self.initialize()

    def initialize(self):
        # initialize LSTM weights and biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)
    
    def forward(self, x, lengths):
        batch_size = x.size()[0]
        total_length = x.size()[1]


        x = x.permute((0,2,1)) # B x F x T
        x = x.unsqueeze(1) # B x C(1) x F x T
        # CNN層を適用する
        # import IPython;IPython.embed();exit()
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x) 
        x = self.cnn4(x)
        x = self.cnn5(x)
        
        x = x.permute((0,3,2,1)) # B x T × F x C
        # CNNの出力をRNNの入力に変形する
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1) # B x T x (F × C)

        # RNN層を適用する
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True)
        # print(x)
        self.lstm.flatten_parameters()
        # import IPython;IPython.embed();exit()
        output, (h_n, c_n) = self.lstm(x, (self.h_init.repeat(1*(1+int(self.bidirectional)), batch_size, 1),
                                self.c_init.repeat(1*(1+int(self.bidirectional)),batch_size,1)))
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length=total_length)
        #print(output.size())
        # output = output.contiguous()
        # output = output.view(-1, output.size()[-1])
        output = self.dropout(output)

        y1 = self.fc_s(output)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(output)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(output)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(output)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4


def Conv2d(c_in,c_out,k_size,padding,act,Mpl,dropout):
    return nn.Sequential(
            nn.Conv2d(c_in,c_out,k_size,padding="same"),
            nn.BatchNorm2d(c_out),
            act,
            nn.MaxPool2d(Mpl),
            nn.Dropout(dropout)
    )

def Conv2dSC(c_in,c_out,k_size,padding,act,dropout):
    return nn.Sequential(
            nn.Conv2d(c_in,c_out,k_size,padding="same"),
            nn.BatchNorm2d(c_out),
            act,
            nn.Dropout(dropout)
    )


def String_Decoder(h_in,h_out):
    return nn.Sequential(
                        Linear(h_in,h_out),
                        nn.LeakyReLU(),
                        Linear(h_out, 5),
                        nn.LogSoftmax(dim = 2))
def Fingering_Decoder(h_in,h_out):
    return nn.Sequential(
                        Linear(h_in,h_out),
                        nn.LeakyReLU(),
                        Linear(h_out, 6),
                        nn.LogSoftmax(dim = 2))
def Position_Decoder(h_in,h_out):
    return nn.Sequential(
                        Linear(h_in,h_out),
                        nn.LeakyReLU(),
                        Linear(h_out, 13),
                        nn.LogSoftmax(dim = 2))
def Bowing_Decoder(h_in,h_out):
    return nn.Sequential(
                        Linear(h_in,h_out),
                        nn.LeakyReLU(),
                        Linear(h_out, 3),
                        nn.LogSoftmax(dim = 2))

# -------TVCG-------

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_feature_size = config['Ginchannels'] # The number of frequency bins of audio: C_in
        self.out_feature_size = config['Goutchannels'] # The output size of Generator (i.e. num_joints * num_quart_dim): C_out
        self.num_input_sequence = 128 # The length of input audio: T

        self.down_res1 = ResBlock(in_channels=self.input_feature_size, out_channels=128)
        self.down_res2 = ResBlock(in_channels=128, out_channels=256)
        self.down_res3 = ResBlock(in_channels=256, out_channels=512)
        self.down_res4 = ResBlock(in_channels=512, out_channels=1024)
        self.down_res5 = ResBlock(in_channels=1024, out_channels=2048)

        # L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        self.pooling = nn.MaxPool1d(kernel_size=2) # Defalut: stride=kernel_size, padding=0, dilation=1

        self.us1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'), # (bs, 2048, T/16) -> (bs, 2048, T/8)
            nn.Conv1d(2048, 1024, kernel_size=1, stride=1), # (bs, 2048, T/8) -> (bs, 1024, T/8)
            nn.BatchNorm1d(num_features=1024), # (bs, 1024. T/8)
            nn.ReLU() # (bs, 1024. T/8)
        )
        self.us2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'), # (bs, 1024, T/8) -> (bs, 1024, T/4)
            nn.Conv1d(1024, 512, kernel_size=1, stride=1), # (bs, 1024, T/4) -> (bs, 512, T/4)
            nn.BatchNorm1d(num_features=512), # (bs, 512, T/4)
            nn.ReLU() # (bs, 512, T/4)
        )
        self.us3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),  # (bs, 512, T/4) -> (bs, 512, T/2)
            nn.Conv1d(512, 256, kernel_size=1, stride=1), # (bs, 512, T/2) -> (bs, 256, T/2)
            nn.BatchNorm1d(num_features=256), #(bs, 256, T/2)
            nn.ReLU() #(bs, 256, T/2)
        )
        self.us4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'), # (bs, 256, T/2) -> (bs, 256, T)
            nn.Conv1d(256, 128, kernel_size=1, stride=1), # (bs, 256, T) -> (bs, 128, T)
            nn.BatchNorm1d(num_features=128), # (bs, 128, T)
            nn.ReLU() # (bs, 128, T)
        )

        self.attb1 = AttentionBlock(num_channels=1024, num_sequence=self.num_input_sequence//8)
        self.attb2 = AttentionBlock(num_channels=512, num_sequence=self.num_input_sequence//4)
        self.attb3 = AttentionBlock(num_channels=256, num_sequence=self.num_input_sequence//2)
        self.attb4 = AttentionBlock(num_channels=128, num_sequence=self.num_input_sequence)

        self.up_res1 = ResBlock(in_channels=2048, out_channels=1024)
        self.up_res2 = ResBlock(in_channels=1024, out_channels=512)
        self.up_res3 = ResBlock(in_channels=512, out_channels=256)
        self.up_res4 = ResBlock(in_channels=256, out_channels=128)

        self.quart_head = nn.Conv1d(in_channels=128, out_channels=self.out_feature_size, kernel_size=1, stride=1) # (bs, 128, T) -> (bs, 248, T)
        # self.quart_norm = nn.BatchNorm1d(self.out_feature_size)

    # def min_max_normalization(self, input, min, max):
    #     orig_min, orig_max = torch.min(input), torch.max(input)
    #     output = (input - orig_min) / (orig_max - orig_min) * (max - min) + min
    #     return output

    def forward(self, x):
        # Down convolutions
        x1_for_skip = self.down_res1(x) # down_res1: (bs, C_in, T) -> (bs, 128, T)
        x_for_down = self.pooling(x1_for_skip) # ds1: (bs, 128, T) -> (bs, 128, T/2)

        x2_for_skip = self.down_res2(x_for_down) # down_res2: (bs, 128, T/2) -> (bs, 256, T/2)
        x_for_down = self.pooling(x2_for_skip) # ds2: (bs, 256, T/2) -> (bs, 256, T/4)

        x3_for_skip = self.down_res3(x_for_down) # down_res3: (bs, 256, T/4) -> (bs, 512, T/4)
        x_for_down = self.pooling(x3_for_skip) # ds3: (bs, 512, T/4) -> (bs, 512, T/8)

        x4_for_skip = self.down_res4(x_for_down) # down_res4: (bs, 512, T/8) -> (bs, 1024, T/8)
        x_for_down = self.pooling(x4_for_skip) # ds4: (bs, 1024, T/8) -> (bs, 1024, T/16)

        x_for_up = self.down_res5(x_for_down) # down_res5: (bs, 1024, T/16) -> (bs, 2048, T/16)
        
        # Up convolutions
        x_for_up = self.us1(x_for_up) # us1: (bs, 2048, T/16) -> (bs, 1024, T/8)
        t_weighted_x = self.attb1(f_d=x4_for_skip, f_u=x_for_up) # attb1: (bs, 1024, T/8) -> (bs, 1024, T/8)
        x_for_up = torch.cat([x_for_up, t_weighted_x], dim=1) # (bs, 1024, T/8) -> (bs, 2048, T/8)
        x_for_up = self.up_res1(x_for_up) # up_res1: (bs, 2048, T/8) -> (bs, 1024, T/8)

        x_for_up = self.us2(x_for_up) # us2: (bs, 1024, T/8) -> (bs, 512, T/4)
        t_weighted_x = self.attb2(f_d=x3_for_skip, f_u=x_for_up) # attb2: (bs, 512, T/4) -> (bs, 512, T/4)
        x_for_up = torch.cat([x_for_up, t_weighted_x], dim=1) # (bs, 512, T/4) -> (bs, 1024, T/4)
        x_for_up = self.up_res2(x_for_up) # up_res2: (bs, 1024, T/4) -> (bs, 512, T/4)

        x_for_up = self.us3(x_for_up) # us3: (bs, 512, T/4) -> (bs, 256, T/2)
        t_weighted_x = self.attb3(f_d=x2_for_skip, f_u=x_for_up) # attb3: (bs, 256, T/2) -> (bs, 256, T/2)
        x_for_up = torch.cat([x_for_up, t_weighted_x], dim=1) # (bs, 256, T/2) -> (bs, 512, T/2)
        x_for_up = self.up_res3(x_for_up) # up_res3: (bs, 512, T/2) -> (bs, 256, T/2)

        x_for_up = self.us4(x_for_up) # us4: ((bs, 256, T/2) -> (bs, 128, T)
        t_weighted_x = self.attb4(f_d=x1_for_skip, f_u=x_for_up) # attb4: (bs, 128, T) -> (bs, 128, T)
        x_for_up = torch.cat([x_for_up, t_weighted_x], dim=1) # (bs, 128, T) -> (bs, 256, T)
        x_for_up = self.up_res4(x_for_up) # up_res4: (bs, 256, T) -> (bs, 128, T)
        out = self.quart_head(x_for_up) # quart_head: (bs, 128, T) -> (bs, 248, T)
        out = out.transpose(2,1).reshape(-1, x.shape[-1], 62, 4)  # shape: (bs, 248, T) -> (bs, T, 62, 4)
        norm = out.norm(p=2, dim=3, keepdim=True) # (bs, T, 62, 4) -> (bs, T, 62, 1)
        out = out / (norm + 1e-6)
        
        
        return out

# Fig. 6
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        self.residual1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1), # (bs, in_channels, T) -> (bs, out_channels, T)
            nn.BatchNorm1d(num_features=out_channels), # (bs, out_channels, T)
            nn.ReLU() # (bs, out_channels, T)
        )
        self.residual2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1), # (bs, out_channels, T) -> (bs, out_channels, T)
            nn.BatchNorm1d(num_features=out_channels), # (bs, out_channels, T)
            nn.ReLU() # (bs, out_channels, T)
        )

    def forward(self, x):
        identity_path = self.conv1(x) # (bs, in_channels, T) -> (bs, out_channels, T)
        residual_path = self.residual2(self.residual1(x)) # (bs, in_channels, T) -> (bs, out_channels, T)
        out = identity_path + residual_path # (bs, out_channels, T)
        return out

# Fig. 8
class AttentionBlock(nn.Module):
    def __init__(self, num_channels, num_sequence=128):
        super().__init__()
        self.wd_linear = nn.Linear(in_features=num_channels, out_features=num_channels, bias=False) # NOT num_sequence!
        self.wu_linear = nn.Linear(in_features=num_channels, out_features=num_channels, bias=False) # NOT num_sequence!

        self.calc_attn_func = nn.Sequential(
            nn.ReLU(), # (bs, num_channels, num_sequence)
            nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1), # (bs, num_channels, num_sequence)  -> (bs, num_channels, num_sequence)
            nn.Sigmoid() # (bs, num_channels, num_sequence)
        )

    def forward(self, f_d, f_u):
        mapped_f_d = self.wd_linear(f_d.transpose(1,2)).transpose(1,2) # (bs, num_channels, num_sequence) -> (bs, num_sequence, num_channels) -> (bs, num_channels, num_sequence)
        mapped_f_u = self.wu_linear(f_u.transpose(1,2)).transpose(1,2) # (bs, num_channels, num_sequence) -> (bs, num_sequence, num_channels) -> (bs, num_channels, num_sequence)
        weights = self.calc_attn_func(mapped_f_d + mapped_f_u) # (bs, num_channels, num_sequence) -> (bs, num_channels, num_sequence)
        out = torch.mul(weights, f_d) # (bs, num_channels, num_sequence)
        return out