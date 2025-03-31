import torch
import torch.nn as nn
from models.uniformer_blocks import uniformer_small
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from typing import List
from torch import Tensor
from utils.ops import aug_rand_with_learnable_rep
import torch.nn.functional as F
from losses.loss import Contrast_Loss

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print ('conv3d: ', x.shape)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class SSLEncoder(nn.Module):
    def __init__(self, num_phase: int, initial_checkpoint: str=None):
        super().__init__()
        self.uniformer = uniformer_small(in_chans=num_phase)
    def forward(self, x):
        x_0, x_enc1, x_enc2, x_enc3, x_enc4 = self.uniformer(x)
        return x_0, x_enc1, x_enc2, x_enc3, x_enc4
 

class UniSegDecoder(nn.Module):
    def __init__(self, img_size: int, in_chans: int,cls_chans=0):
        super().__init__()
        self.decoder5 = UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=512,
                    out_channels=320,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name="instance",
                    res_block=True,
                )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=320,
            out_channels=128,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.proj1 = PatchEmbed(
                img_size=img_size, patch_size=3, in_chans=in_chans, embed_dim=64, stride=1, padding=1)    
        
        # NOTE in ds this part is replaced with decoder2
        # self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        if cls_chans==0:
            self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=in_chans)
        else:
            self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=cls_chans)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x0, x1, x2, x3, x4):
        # we do not use skip connection for better representation learning
        
        dec5 = self.decoder5.transp_conv(x4.permute(0,1,3,4,2))
        dec4 = self.decoder4.transp_conv(dec5) 
        dec3 = self.decoder3.transp_conv(dec4)
        dec2 = self.decoder2.transp_conv(dec3)
        x_rec = self.out_1(dec2) # 4

        return x_rec


    
class RecModel(nn.Module):
    def __init__(self, args, dim=768):
        super(RecModel, self).__init__()
        self.device = args.device
        in_chans = args.in_channels
        img_size = args.roi_x
        self.encoder = SSLEncoder(num_phase=in_chans, initial_checkpoint=args.initial_checkpoint)
        self.decoder = UniSegDecoder(img_size=img_size, in_chans=in_chans)
        num_modals = len(args.template_index)
        # we found use zero-init will accelerate template learning and produce better visiability
        self.rep_template = nn.Parameter(torch.zeros((num_modals,args.dst_h, args.dst_w, args.dst_d)))
        
        self.kl_loss = Contrast_Loss(temperature=1.0)
        self.recon_loss = nn.MSELoss(reduction="mean")
        
    def forward(self, x:Tensor,clip_text_feat=None,ret_med=False,
                args=None,index=-1,rep_start_point=-1,rearranged_series=None,
                cur_epoch=-1,series_name=None):
        
        rep_template = self.rep_template  
        B = x.shape[0]
        # mask with template
        x_aug_i_1 = aug_rand_with_learnable_rep(args,x,rep_template,index=index,rep_start_point=rep_start_point,rearranged_series=rearranged_series,series_name=series_name,use_temp=True)   
        flag = cur_epoch<=300
        # mask with the other modality <300 for faster template learning
        x_aug_i_2 = aug_rand_with_learnable_rep(args,x,rep_template,index=index,rep_start_point=rep_start_point,rearranged_series=rearranged_series,series_name=series_name,use_temp=flag)   

        x_aug = torch.cat((x_aug_i_1,x_aug_i_2),dim=0)       
        feature = self.encoder(x_aug)
        
        feature_x4 = feature[-1]
        teacher = feature_x4[:B]
        student = feature_x4[B:]
        
        # contrastive learning is introduced when template is stable
        if cur_epoch>=args.start_epoch:
            y_log = teacher.flatten(1)
            x_log = student.flatten(1)
            loss_cons = self.kl_loss(x_log,y_log) #change to Contrastive loss 
            loss_cons_item = loss_cons.item()
        else:
            loss_cons = 0.0
            loss_cons_item = 0.0
        # recons
        rec_x1_1 = self.decoder(feature[0],feature[1],feature[2],feature[3],feature[4])
        target = x[:,index,:,:,:].unsqueeze(1)
        
        loss_rec_1 = self.recon_loss(rec_x1_1[:B],target)
        loss_rec_2 = self.recon_loss(rec_x1_1[B:],target)

        loss = loss_rec_1*10  + loss_rec_2+loss_cons
        # if args.rank==0:print(f"loss_total:{loss.item():5f}, loss_rec_1:{loss_rec_1.item()*10:5f}, loss_rec_2:{loss_rec_2.item():5f}")
        return loss,loss_rec_1.item(),loss_rec_2.item(),loss_cons_item