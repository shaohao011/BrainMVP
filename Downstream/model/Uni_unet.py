import torch.nn as nn
from .uniformer import uniformer_small, PatchEmbed
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock


class UniSegDecoder(nn.Module):
    def __init__(self, img_size: int, in_chans: int, cls_chans=0, segmentation=False):
        super().__init__()
        self.segmentation = segmentation
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
        
        if not self.segmentation:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        if cls_chans==0:
            self.out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=in_chans)
        else:
            self.out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=cls_chans)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x0, x1, x2, x3, x4):
        dec5 = self.decoder5(x4.permute(0,1,3,4,2), x3.permute(0,1,3,4,2))
        dec4 = self.decoder4(dec5, x2.permute(0,1,3,4,2)) #128
        dec3 = self.decoder3(dec4, x1.permute(0,1,3,4,2)) # convert to C,H,W,D
        if self.segmentation:
            x_proj = self.proj1(x0)
            dec2 = self.decoder2(dec3, x_proj.permute(0,1,3,4,2))
            x_out = self.out(dec2)
            return dec5, dec4, dec3, dec2, x_out
        x_up = self.up(dec3) # 64
        x_out = self.out(x_up) # 4
        return dec5, dec4, dec3, x_up, x_out


class UniUnet(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=64, p=0.2, multi_scale = False, segmentation=True):
        super(UniUnet, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ms = multi_scale

        self.ms_out = []
        self.up_out = []

        self.dropout = nn.Dropout(p=p)

        self.encoder = uniformer_small(img_size=self.input_shape, in_chans=self.in_channels)
        self.decoder = UniSegDecoder(img_size=self.input_shape, in_chans=self.in_channels, cls_chans=self.out_channels, segmentation=segmentation)
        
        if self.ms:
            self.ms_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            
            self.ms_out.append(nn.Conv3d(init_channels*1, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            
            self.ms_out = nn.ModuleList(self.ms_out)

        
    def forward(self, x, location=None):
        x0, x1, x2, x3, x4 = self.encoder(x)
        s5, s4, s3, s2, out = self.decoder(x0, x1, x2, x3, x4)

        if self.ms and self.training:
        
            out4 = self.up_out[0](self.ms_out[0](s4))
            out3 = self.up_out[1](self.ms_out[1](s3))
            out = [out4, out3, out]

            return out, out4
        
        if self.training:
            return out, x4
        
        return out

