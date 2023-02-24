import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from models.gcn import GraphConvolution

class GCN(nn.Module):
    def __init__(self, in_channels, hidden,out_channels, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channels,in_channels)
        self.gc2 = GraphConvolution(in_channels, hidden)
        self.gc3 = GraphConvolution(hidden, out_channels)
        self.relu = F.relu
        self.Fdropout = F.dropout
        self.dropout =dropout

    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc2(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc3(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        return x

class EdgeExpand(nn.Module):   
    def __init__(self,channels):
        super(EdgeExpand,self).__init__()
        kernel_x = [[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]
        kernel_y = [[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(channels,channels,3,3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(channels,channels,3,3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = F.softmax
    
    def forward(self,x):
        sobel_x = F.conv2d(x,self.weight_x,stride=1, padding=1)
        #sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x,self.weight_y,stride=1, padding=1)
        #sobel_y = torch.abs(sobel_y)
        sobel = sobel_x+sobel_y
        #在channel维度上进行softmax
        sobel = self.softmax(sobel,dim=1)
        out = x*sobel
        return out


class gen_adj(nn.Module):
    def __init__(self, in_channels):
        super(gen_adj, self).__init__()
        if in_channels==1:
             self.adj_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )
        else:
            self.adj_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//4, in_channels//8, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels//8),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//8, in_channels//16, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels//16),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//16, 1, kernel_size=3, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
                )
        self.projection=nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
            )
    def forward(self, x):
        b,c,h,w=x.size()
        x = self.adj_conv(x)
        out_adj_a = x.view(b,1,-1)
        out_adj_b = x.view(b,-1,1)
        out_adj = torch.bmm(out_adj_b,out_adj_a)
        out_adj = torch.unsqueeze(out_adj,1)
        out_adj = self.projection(out_adj)
        out_adj = torch.squeeze(out_adj,1)
        return out_adj



class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #边缘增强
        self.expand_edge = EdgeExpand(out_channels)
        # ceil_mode参数取整的时候向上取整，该参数默认为False表示取整的时候向下取整
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out = self.down_conv(x)
        out_expand = self.expand_edge(out)
        out_pool = self.pool( out_expand)
        return out, out_pool


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 反卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # 迭代代替填充， 取得更好的结果
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # 如果填充物体积大小不同
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # 连接
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parametersL {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\nNbr of trainable parameters: {nbr_params}"


class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=5, freeze_bn=False, **_):
        super(UNet, self).__init__()

        #unet编码
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        #gcn编码
        self.gcn_encoder=GCN(1024,2048,1024,0.5)
        self.adj_encoder=gen_adj(1024)
        # #gcn解码
        # self.gcn_decoder=GCN(1,512,1024,0.5)
        # self.adj_decoder=gen_adj(1)
        #unt解码
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)

        #全连接
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initalize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):

        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)

        adj_en =self.adj_encoder(x)
        x=self.gcn_encoder(x,adj_en)

        # adj_de=self.adj_decoder(x)
        # x=self.gcn_decoder(x,adj_de)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()