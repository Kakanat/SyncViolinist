import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True):
        """
        Args:
            in_dim: dimension of input
            out_dim: dimension of output
            bias: boolean. if True, bias is included.
        """
        super(Linear, self).__init__() 
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

#        nn.init.uniform_(self.linear.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.linear.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)

class Conv1d(nn.Module):
    """
    Convolution 1d Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                padding=0, dilation=1, bias=True):
        """
        Args:
            in_channels: dimension of input
            out_channels: dimension of output
            kernel_size: size of kernel
            stride: size of stride
            padding: size of padding
            dilation: dilation rate
            bias: boolean. if True, bias is included.
        """
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation,
                            bias=bias)

#        nn.init.uniform_(self.conv.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.conv.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """
    Convolution block which is used in U-net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.residual = residual
        self.double_conv = nn.Sequential(
            Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        if self.residual: 
            self.bypass = nn.Sequential(
                Conv1d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        if self.residual:
            return F.relu(self.double_conv(x) + self.bypass(x))
        else:
            return F.relu(self.double_conv(x))


class Down(nn.Module):
    """
    Downscaling with avgpool then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Down, self).__init__()      
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv(in_channels, out_channels, residual=residual),
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Up(nn.Module):
    """
    Upscaling by linear interpotation then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Up, self).__init__()       
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual=residual)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True):
        """
        Args:
            in_dim: dimension of input
            out_dim: dimension of output
            bias: boolean. if True, bias is included.
        """
        super(Linear, self).__init__() 
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

#        nn.init.uniform_(self.linear.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.linear.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)

class Conv1d(nn.Module):
    """
    Convolution 1d Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                    padding=0, dilation=1, bias=True):
        """
        Args:
            in_channels: dimension of input
            out_channels: dimension of output
            kernel_size: size of kernel
            stride: size of stride
            padding: size of padding
            dilation: dilation rate
            bias: boolean. if True, bias is included.
        """
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=bias)

#        nn.init.uniform_(self.conv.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.conv.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """
    Convolution block which is used in U-net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.residual = residual
        self.double_conv = nn.Sequential(
            Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        if self.residual: 
            self.bypass = nn.Sequential(
                Conv1d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        if self.residual:
            return F.relu(self.double_conv(x) + self.bypass(x))
        else:
            return F.relu(self.double_conv(x))


class Down(nn.Module):
    """
    Downscaling with avgpool then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Down, self).__init__()      
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv(in_channels, out_channels, residual=residual),
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Up(nn.Module):
    """
    Upscaling by linear interpotation then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Up, self).__init__()       
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual=residual)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# --- TCN ---

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        #self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        #self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class FBClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout = 0.2):
        super(FBClassifier, self).__init__()
        
        hidden_dim_1 = input_dim
        hidden_dim_2 = hidden_dim

        self.fc_s = nn.Sequential(
                                Linear(hidden_dim_1, hidden_dim_2),
                                nn.LeakyReLU(),
                                nn.Dropout(dropout),
                                Linear(hidden_dim_2, 5),
                                nn.LogSoftmax(dim = 2))
        self.fc_f = nn.Sequential(
                                Linear(hidden_dim_1, hidden_dim_2),
                                nn.LeakyReLU(),
                                nn.Dropout(dropout),
                                Linear(hidden_dim_2, 6),
                                nn.LogSoftmax(dim = 2))
        
        self.fc_p = nn.Sequential(
                                Linear(hidden_dim_1, hidden_dim_2),
                                nn.LeakyReLU(),
                                nn.Dropout(dropout),
                                Linear(hidden_dim_2, 13),
                                nn.LogSoftmax(dim = 2))
    
        self.fc_u = nn.Sequential(
                                Linear(hidden_dim_1, hidden_dim_2),
                                nn.LeakyReLU(),
                                nn.Dropout(dropout),
                                Linear(hidden_dim_2, 3),
                                nn.LogSoftmax(dim = 2))
        self.initialize()
        
    def initialize(self):
        for fc in [self.fc_s, self.fc_f, self.fc_p, self.fc_u]:
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
    def forward(self, x):
        y1 = self.fc_s(x)
        y1 = y1.permute((0,2,1))
        y2 = self.fc_f(x)
        y2 = y2.permute((0,2,1))
        y3 = self.fc_p(x)
        y3 = y3.permute((0,2,1))
        y4 = self.fc_u(x)
        y4 = y4.permute((0,2,1))
        return y1, y2, y3, y4


