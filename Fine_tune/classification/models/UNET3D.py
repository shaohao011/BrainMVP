import torch.nn as nn
import torch

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
    
    
class UNet3D_g(nn.Module):
    """
    A normal 3D - Unet, different from the original model architecture in Ref, which use the content and style to reconstruc the high level feature.
    
    3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False):
        super(UNet3D_g, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ds = deep_supervised
        
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)
        

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

    def forward(self, x):
        c1 = self.conv1a(x) # (B, 4, 128, 128, 128) -> (B, 16, 128, 128, 128)
        c1 = self.conv1b(c1)    # identical
        c1d = self.ds1(c1)  # (B, 16, 128, 128, 128) -> (B, 32, 64, 64, 64)
        
        c2 = self.conv2a(c1d)   # identical
        c2 = self.conv2b(c2)    # identical
        c2d = self.ds2(c2)  # (B, 32, 64, 64, 64) -> (B, 64, 32, 32, 32)
        c2d_p = self.pool(c2d)  # kernel: 2; (B, 64, 32, 32, 32) -> (B, 64, 16, 16, 16) 作用？
        
        c3 = self.conv3a(c2d)   # identical
        c3 = self.conv3b(c3)    # identical
        c3d = self.ds3(c3)  # (B, 64, 32, 32, 32) -> (B, 128, 16, 16, 16)

        c4 = self.conv4a(c3d)  # identical
        c4 = self.conv4b(c4)    # identical
        c4 = self.conv4c(c4)    # identical
        c4d = self.conv4d(c4)   # identical, (B, 128, 16, 16, 16)

        style = [c2d, c3d, c4d]
        content = c4d 
        
        u4 = self.up4conva(c4d) # (B, 128, 16, 16, 16) -> (B, 64, 16, 16, 16)
        u4 = self.up4(u4)    # -> (B, 64, 32, 32, 32)
        u4 = u4 + c3
        u4 = self.up4convb(u4)  # identical

        u3 = self.up3conva(u4)  # -> (B, 32, 32, 32, 32)
        u3 = self.up3(u3)   # -> (B, 32, 64, 64, 64)
        u3 = u3 + c2        
        u3 = self.up3convb(u3)  # identical

        u2 = self.up2conva(u3)  # (B, 16, 64, 64, 64)
        u2 = self.up2(u2)   # (B, 16, 128, 128, 128)
        u2 = u2 + c1   
        u2 = self.up2convb(u2)  # identical

        uout = self.up1conv(u2) # (B, 4, 128, 128, 128)
        
        
        if self.ds and self.training:
        
            out4 = self.up_out[0](self.ds_out[0](u4))
            out3 = self.up_out[1](self.ds_out[1](u3))
            uout = [out4, out3, uout]
        
        return uout, style, content

# NOTE: used for basline
class UNet3D_gb(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False):
        super(UNet3D_gb, self).__init__()
        self.unet = UNet3D_g(input_shape, in_channels, out_channels, init_channels, p, deep_supervised = deep_supervised)
    def forward(self, x, location = None):
        uout, style, content = self.unet(x)
        # if self.training:
        #     return uout, style, content
        return uout
    

class cls_model(torch.nn.Module):
    def __init__(self,in_channels, out_channels, img_size,  init_channels=32, pretrain_path='',deep_supervised=False):
        super().__init__()
        #==============Model====================
        self.model  = UNet3D_gb(input_shape=img_size, 
                      in_channels=in_channels,
                      out_channels=out_channels, 
                      init_channels=init_channels, 
                      deep_supervised=deep_supervised)
        if pretrain_path:
            print("[!]using pretrain model")
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            state_dict = checkpoint['state_dict']
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") 
            del checkpoint
            del state_dict['conv1a.weight']
            del state_dict['rep_template']
            del state_dict['conv1a.bias']
            del state_dict['up1conv.weight']
            del state_dict['up1conv.bias']
            del state_dict
            # Load pretrain model here
            self.model.unet.load_state_dict(state_dict, strict=False)
        #=======================================
        self.head = torch.nn.Linear(256,out_channels)
        
    def forward(self,x):
        c1 = self.model.unet.conv1a(x) # (B, 4, 128, 128, 128) -> (B, 16, 128, 128, 128)
        c1 = self.model.unet.conv1b(c1)    # identical
        c1d = self.model.unet.ds1(c1)  # (B, 16, 128, 128, 128) -> (B, 32, 64, 64, 64)
        
        c2 = self.model.unet.conv2a(c1d)   # identical
        c2 = self.model.unet.conv2b(c2)    # identical
        c2d = self.model.unet.ds2(c2)  # (B, 32, 64, 64, 64) -> (B, 64, 32, 32, 32)
        c2d_p = self.model.unet.pool(c2d)  # kernel: 2; (B, 64, 32, 32, 32) -> (B, 64, 16, 16, 16) 作用？
        
        c3 = self.model.unet.conv3a(c2d)   # identical
        c3 = self.model.unet.conv3b(c3)    # identical
        c3d = self.model.unet.ds3(c3)  # (B, 64, 32, 32, 32) -> (B, 128, 16, 16, 16)

        c4 = self.model.unet.conv4a(c3d)  # identical
        c4 = self.model.unet.conv4b(c4)    # identical
        c4 = self.model.unet.conv4c(c4)    # identical
        c4d = self.model.unet.conv4d(c4)   # identical, (B, 128, 16, 16, 16)
        out = self.head(c4d.flatten(2).mean(-1))
        return out

def make_model(in_channels, out_channels, img_size, feature_size,pretrain_path=None):
    model = cls_model(in_channels, out_channels, img_size, feature_size,pretrain_path)
    return model