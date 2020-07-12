import      torch
nn          = torch.nn
F           = nn.functional
from        torch                           import  optim
from        torchvision                     import  utils as vutils
from        torch                           import  autograd

from        attr_dict                       import  *
from        utils                           import  *
from        torchvision.models.resnet       import  BasicBlock, Bottleneck
import      torchvision.models              as      vmodels

import      matplotlib.pyplot               as      plt
import      os
import      numpy                           as      np
import      sys
import      imageio
from        contextlib                      import  ExitStack
import      random

def get_multiplier_for_feats_net(options):
    model_arch              = options.model_arch
    if any([x in model_arch for x in ['ResNet18','ResNet34']]):
        multiplier          = 8
    elif any([x in model_arch for x in ['ResNet50','ResNet101','ResNet152']]):
        multiplier          = 32

    return multiplier


def get_multiplier(options):
    model_arch              = options.model_arch
    multiplier              = get_multiplier_for_feats_net(options)

    return multiplier



# ===============================================================================================================================
#   Convolution functions
def conv5x5(in_planes, out_planes, stride=1, bias=False):
    '''5x5 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    '''3x3 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def weights_init(m, nonlinearity='relu'):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
# ===============================================================================================================================


def weights_init2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# ===============================================================================================================================


resnet_inits            = {
        10  :       lambda flag: vmodels.ResNet(BasicBlock, [1, 1, 1, 1]),
        14  :       lambda flag: vmodels.ResNet(BasicBlock, [1, 1, 2, 2]),
        18  :       lambda flag: vmodels.resnet18(pretrained=flag),
        34  :       lambda flag: vmodels.resnet34(pretrained=flag),
        50  :       lambda flag: vmodels.resnet50(pretrained=flag),
        101 :       lambda flag: vmodels.resnet101(pretrained=flag),
        152 :       lambda flag: vmodels.resnet152(pretrained=flag),
}

possible_resnets                        = resnet_inits.keys()
resnet_outsize                          = {
        10  :       512, 
        14  :       512, 
        18  :       512, 
        34  :       512, 
        50  :       2048, 
        101 :       2048, 
        152 :       2048,
}


def ResNet18(options):
    return ResNetPretrained(options, 18)

def ResNet34(options):
    return ResNetPretrained(options, 34)

def ResNet50(options):
    return ResNetPretrained(options, 50)

def ResNet101(options):
    return ResNetPretrained(options, 101)

def ResNet152(options):
    return ResNetPretrained(options, 152)

class ResNetPretrained(nn.Module):
    def __init__(self, options, depth):
        super(ResNetPretrained, self).__init__()

        self.conv_block         = resnet_inits[depth](True)

        self.expansion          = 1 if depth < 50 else 4
        self.f_size             = 64 * 8 * self.expansion

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv_block.conv1(x)
        x = self.conv_block.bn1(x)
        x = self.conv_block.relu(x)
        h0 = self.conv_block.maxpool(x)

        h1 = self.conv_block.layer1(h0)
        h2 = self.conv_block.layer2(h1)
        h3 = self.conv_block.layer3(h2)
        h4 = self.conv_block.layer4(h3)

        if hasattr(kwargs, 'return_h_feats') and kwargs['return_h_feats']:
            return_dict['feats_0']  = h0
            return_dict['feats_1']  = h1
            return_dict['feats_2']  = h2
            return_dict['feats_3']  = h3
        return_dict['fg_feats'] = h4
        return return_dict


# ===============================================================================================================================

class DilatedConvBlock(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConvBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.main(x)

class DilatedConv2ResBlock(nn.Module):
    """
    Residual convolutional block with dilated filters. 
    """
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConv2ResBlock, self).__init__()

        self.in_c   = in_c
        self.out_c  = out_c

        # Residual connection
        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
        )

        if out_c != in_c:
            # Mapping connection. 
            self.mapper = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_c),
            )
        self.relu   = nn.ReLU(True)

    def forward(self, x, **kwargs):
        residual        = self.res(x)
        if self.in_c != self.out_c:
            x           = self.mapper(x)

        out             = self.relu(x + residual)
        return out


class LinearScaleClassifier(nn.Module):
    """
    A simple classifier based on average pooling followed by a linear function. 
    """
    def __init__(self, options):
        super(LinearScaleClassifier, self).__init__()

        # Get the size of the hidden space---number of channels. 
        multiplier              = get_multiplier(options)
        self.f_size             = options.ndf * multiplier

        # Number of targets is the number of levels we are training with. 
        self.n_targets          = len(options.levels)
       
        # Average pooler
        self.avg_pool           = nn.AdaptiveAvgPool2d((1, 1))
        # Linear classifier. 
        self.linear             = nn.Linear(self.f_size, self.n_targets)

    def forward(self, fg_feats, **kwargs):
        batch_size              = fg_feats.size(0)
        x                       = self.avg_pool(fg_feats).view(batch_size, -1)
        x                       = self.linear(x)
        return x


class Dilated10ConvAttentionMap1x1AvgTauResNet34WithSparsity2(nn.Module):
    """
    Combines the attention network F and the feature extractor of the ResNet34 G. 
    attention_net is the feature extractor based on residual blocks of dilated
    convolutions, which regresses a confidence map of the same size as the input
    image. 

    feats_net is the feature extractor ResNet34. 
    """
    def __init__(self, options):
        super(Dilated10ConvAttentionMap1x1AvgTauResNet34WithSparsity2, self).__init__()

        self.nc             = options.nc
        self.ndf            = options.attention_feats_ex_ndf

        self.attention_net  = nn.Sequential(
            DilatedConvBlock(self.nc, self.ndf, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=10, dilation=10, bias=False),

            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=20, dilation=20, bias=False),

            nn.Conv2d(self.ndf, 1, kernel_size=1, stride=1, padding=0),
        )

        self.feats_net      = ResNet34(options)


        # self.first records whether the first iteration is over or not. If it is, then the inferred tau
        #   from the training batch is used to set self.tau.
        # Otherwise, self.tau = 0.9 * self.tau + 0.1 * self.tau_new is used. 
        self.register_buffer('first', torch.ByteTensor([1]))
        # The learnt threshold for the compressed sigmoid. 
        self.register_buffer('tau', torch.FloatTensor([0.]))
        # Required sparsity
        self.p              = options.attention_sparsity
        # Compression of the sigmoid. 
        self.r              = options.attention_sparsity_r
        # The compressed and biased sigmoid
        self.sigmoid        = lambda x: torch.sigmoid(self.r * (x))

    def forward(self, images, train=True, attention_only=False, **kwargs):
        return_dict         = {}

        # Get the confidence map. 
        attention           = self.attention_net(images)
        
        if train:
            # In train phase, use the tau obtained from this training batch to 
            # compute the threshold. 
            A               = attention.detach().view(attention.size(0), -1).contiguous()
            A, _            = torch.sort(A, dim=1, descending=True)
            t_idx           = int(np.floor(self.p * A.size(1)))
            tau             = torch.mean(A[:, t_idx]).item()

            # If no training batches have been seen so far. 
            if self.first.item():
                self.tau.fill_(tau)
                self.first.fill_(0)
            # Else, use the following formula to update self.tau
            else:
                self.tau    = 0.9 * self.tau + 0.1 * tau

        else:
            # In the testing phase, use the learnt tau. 
            tau             = self.tau

        # Activate the confidence maps using the sigmoid. 
        attention           = self.sigmoid(attention - tau)
        return_dict['attention']    = attention

        if attention_only:
            return_dict['fg_feats'] = None
            return return_dict


        # Pass the masked image to the feature extractor. 
        feats_out           = self.feats_net(images * attention)

        for k in feats_out:
            return_dict[k]          = feats_out[k]

        return return_dict

