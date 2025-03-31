#!/usr/bin/env python3
# encoding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn as nn
from torch import Tensor
from utils.ops import aug_rand_with_learnable_rep
import torch.nn.functional as F
from losses.loss import Contrast_Loss

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

class encoder(nn.Module):
    def __init__(self, init_channels=8):
        super(encoder, self).__init__()
        self.init_channels = init_channels
        self.conv1a = nn.Conv3d(1, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels*1)  # 32

        self.ds1 = torch.nn.MaxPool3d(2)
                    #nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                            # padding=(1, 1, 1))  # down sampling and add channels

        #self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 1, init_channels * 2)

        self.ds2 = torch.nn.MaxPool3d(2)
        #nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        #self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 2, init_channels * 4)

        self.ds3 = torch.nn.MaxPool3d(2)
        #nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        #self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 4, init_channels * 8)
        
        #self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        #self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def forward(self, x):
        
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        
        #c2 = self.conv2a(c1d)
        c2 = self.conv2b(c1d)
        c2d = self.ds2(c2)
        
        #c3 = self.conv3a(c2d)
        c3 = self.conv3b(c2d)
        c3d = self.ds3(c3)

        #c4 = self.conv4a(c3d)
        c4 = self.conv4b(c3d)
        #c4 = self.conv4c(c4)
        #c4d = self.conv4d(c4)

        return [c1, c2, c3, c4]

class decoder(nn.Module):
    def __init__(self, init_channels=8):
        super(decoder, self).__init__()
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 8, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 4, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels*2, init_channels)
        
        self.pool     = nn.MaxPool3d(kernel_size = 2)
        #self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        #self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, 3, (1, 1, 1))

    def forward(self, x):
        for l in x:
            print(l.shape)
        u4 = self.up4conva(x[3])
        u4 = self.up4(u4)
        u4 = torch.cat([u4 , x[2]], 1)
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = torch.cat([u3 , x[1]], 1)
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = torch.cat([u2 , x[0]], 1)
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)

        return uout


class RecModel(nn.Module):

    def __init__(self,args,dim=-1):
        super(RecModel, self).__init__()
        input_shape = (args.roi_x)*3
        in_channels = args.in_channels
        out_channels=1
        init_channels=32 #16 default
        p=0.2
        deep_supervised = False
        
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ds = deep_supervised
        
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)
        
        num_modals = len(args.template_index)
        self.rep_template = nn.Parameter(torch.zeros((num_modals,args.dst_h, args.dst_w, args.dst_d)))
        self.kl_loss = Contrast_Loss(temperature=1.0) 
        self.recon_loss = nn.MSELoss(reduction="mean")
        

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)
        
        self.pool     = nn.MaxPool3d(kernel_size = 2)
        #self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        #self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))
        
        self.ds_out = []
        self.up_out = []
        if self.ds:
            self.ds_out.append(nn.Conv3d(init_channels*4, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            
            self.ds_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            
            self.ds_out = nn.ModuleList(self.ds_out)

    # def forward(self, x):
    def encoder(self,x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)
        return [c1d,c2d,c3d,c4d]
    
    def decoder(self,x):
        c1d,c2d,c3d,c4d = x
        
        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        # u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        # u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        # u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        return uout
    
    def forward(self, x:Tensor,clip_text_feat=None,ret_med=False,args=None,index=-1,rep_start_point=-1,rearranged_series=None,cur_epoch=-1,series_name=None):
        
        rep_template = self.rep_template  
        
        B = x.shape[0]
        x_aug_i_1 = aug_rand_with_learnable_rep(args,x,rep_template,index=index,rep_start_point=rep_start_point,rearranged_series=rearranged_series,series_name=series_name,use_temp=True)   
        flag = cur_epoch<=300
        x_aug_i_2 = aug_rand_with_learnable_rep(args,x,rep_template,index=index,rep_start_point=rep_start_point,rearranged_series=rearranged_series,series_name=series_name,use_temp=flag)   
        x_aug = torch.cat((x_aug_i_1,x_aug_i_2),dim=0)  
        feature = self.encoder(x_aug)
        feature_x4 = feature[-1]
        teacher = feature_x4[:B]
        student = feature_x4[B:]
        if cur_epoch>=args.start_epoch:
            y_log = teacher.flatten(1)
            x_log = student.flatten(1)
            loss_cons = self.kl_loss(x_log,y_log) 
            loss_cons_item = loss_cons.item()
        else:
            loss_cons = 0.0
            loss_cons_item = 0.0
        # recons
        rec_x1_1 = self.decoder((feature[0],feature[1],feature[2],feature[3]))
        target = x[:,index,:,:,:].unsqueeze(1)
        target_x = target
        loss_rec_1 = self.recon_loss(rec_x1_1[:B],target_x)
        loss_rec_2 = self.recon_loss(rec_x1_1[B:],target_x)
        loss = loss_rec_1*10  + loss_rec_2+loss_cons 
        # if args.rank==0:print(f"loss_total:{loss.item():5f}, loss_rec_1:{loss_rec_1.item()*10:5f}, loss_rec_2:{loss_rec_2.item():5f}")
        return loss,loss_rec_1.item(),loss_rec_2.item(),loss_cons_item