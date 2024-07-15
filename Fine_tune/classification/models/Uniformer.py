import torch.nn as nn
import torch
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W )
        return x        
   

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        # print (in_chans, embed_dim, stride)
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
    
    
class UniFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2))

        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=2, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(2, 2, 2))      ###wll 20230711 modify stride
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(2, 2, 2))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])
        self.norm = nn.BatchNorm3d(embed_dim[-1])
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # if isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        # print ('input x: ', x.shape)
        x = x.permute(0,1,4,2,3)      #### change C*H*W*D to C*D*H*W

        x1 = self.patch_embed1(x)
        x1 = self.pos_drop(x1)
        for blk in self.blocks1:
            x1 = blk(x1)
        # print ('1. x: ', x1.shape)

        x2 = self.patch_embed2(x1)
        for blk in self.blocks2:
            x2 = blk(x2)
        # print ('2. x: ', x2.shape)

        x3 = self.patch_embed3(x2)
        for blk in self.blocks3:
            x3 = blk(x3)
        # print ('3. x: ', x3.shape)

        x4 = self.patch_embed4(x3)
        for blk in self.blocks4:
            x4 = blk(x4)
        # print ('4. x: ', x4.shape)
        x4 = self.norm(x4)

        return x, x1, x2, x3, x4

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     # x = x.flatten(2).mean(-1)
    #     # x = self.head(x)
    #     return x

def uniformer_small(**kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
    # model.default_cfg = _cfg()
    return model

# def uniformer_xxs(pretrained=True, **kwargs):
#     model = UniFormer(
#         depth=[2, 5, 8, 2],
#         embed_dim=[56, 112, 224, 448], head_dim=28, mlp_ratio=3, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


# def uniformer_small_plus(pretrained=True, **kwargs):
#     model = UniFormer(
#         depth=[3, 5, 9, 3], conv_stem=True,
#         embed_dim=[64, 128, 320, 512], head_dim=32, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model

# def uniformer_small_plus_dim64(pretrained=True, **kwargs):
#     model = UniFormer(
#         depth=[3, 5, 9, 3], conv_stem=True,
#         embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model

def uniformer_base(pretrained=True, **kwargs):
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# def uniformer_base_ls(pretrained=True, **kwargs):
#     global layer_scale
#     layer_scale = True
#     model = UniFormer(
#         depth=[5, 8, 20, 7],
#         embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
    


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
        # 3 skip-connection
        dec5 = self.decoder5(x4.permute(0,1,3,4,2), x3.permute(0,1,3,4,2))
        dec4 = self.decoder4(dec5, x2.permute(0,1,3,4,2)) #128
        # dec3 = self.decoder3.transp_conv(dec4)#, x1.permute(0,1,3,4,2)) #64
        dec3 = self.decoder3(dec4, x1.permute(0,1,3,4,2)) # convert to C,H,W,D
        if self.segmentation:
            x_proj = self.proj1(x0)
            
            dec2 = self.decoder2(dec3, x_proj.permute(0,1,3,4,2))
            x_out = self.out(dec2)
            return dec5, dec4, dec3, dec2, x_out
            # return x_out
        x_up = self.up(dec3) # 64
        x_out = self.out(x_up) # 4
        return dec5, dec4, dec3, x_up, x_out


class UniUiformer(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False, segmentation=False):
        super(UniUiformer, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ds = deep_supervised

        self.ds_out = []
        self.up_out = []

        self.dropout = nn.Dropout(p=p)

        self.encoder = uniformer_small(img_size=self.input_shape, in_chans=self.in_channels)
        self.decoder = UniSegDecoder(img_size=self.input_shape, in_chans=self.in_channels, cls_chans=self.out_channels, segmentation=segmentation)
        
        if self.ds:
            self.ds_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            
            self.ds_out.append(nn.Conv3d(init_channels*1, self.out_channels, (1, 1, 1)))
            # self.ds_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            
            self.ds_out = nn.ModuleList(self.ds_out)

        
    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        style = [x2, x3, x4]
        content = x4
        
        u5, u4, u3, u2, uout = self.decoder(x0, x1, x2, x3, x4)

        if self.ds and self.training:
        
            out4 = self.up_out[0](self.ds_out[0](u4))
            out3 = self.up_out[1](self.ds_out[1](u3))
            uout = [out4, out3, uout]
        
        return uout, style, content


class Uniformer_b(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False, segmentation=False):
        super(Uniformer_b, self).__init__()
        self.uniformer = UniUiformer(input_shape, in_channels, out_channels, init_channels, p, deep_supervised = deep_supervised, segmentation=segmentation)
    def forward(self, x, location = None):
        uout, style, content = self.uniformer(x)
        # if self.training:
        #     return uout, style, content
        return uout
    
class cls_model(torch.nn.Module):
    def __init__(self,in_channels, out_channels, img_size,  init_channels=16, pretrain_path='',deep_supervised=False):
        super().__init__()
        #==============Model====================
        self.model  = Uniformer_b(input_shape=img_size,
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    init_channels=init_channels,
                    deep_supervised = deep_supervised, 
                    segmentation=True)
        if pretrain_path:
            print("[!]using pretrain model")
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            state_dict = checkpoint['state_dict']
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") 
            del checkpoint
            new_dict ={}
            for k,v in state_dict.items():
                if "uniformer" in k:
                    new_k = "uniformer.encoder."+k[18:]
                    new_dict[new_k] = v
            del state_dict
            del new_dict['uniformer.encoder.patch_embed1.proj.weight']
            self.model.load_state_dict(new_dict, strict=False)
            
            # Load pretrain model here
        #=======================================
        self.head = torch.nn.Linear(512,out_channels)
        
    def forward(self,x):
        _,_,_,_,x4 = self.model.unet.encoder(x) # 
        out = self.head(x4.flatten(2).mean(-1))
        return out

def make_model(in_channels, out_channels, img_size, feature_size,pretrain_path=None):
    model = cls_model(in_channels, out_channels, img_size, feature_size,pretrain_path)
    return model