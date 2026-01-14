# model/swin_unetr_unimatch.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer, WindowAttention
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class SwinUNETR_Seg(nn.Module):
    """
    EchoCare-style SwinUNETR (2D) segmentation backbone:
    - encoder: SwinTransformer (in_chans=3)
    - decoder: UNETR-style blocks
    """
    def __init__(
        self,
        seg_num_classes: int,
        ssl_checkpoint: str = None,
        in_chans: int = 3,
        r: int = 5
    ):
        super().__init__()

        # Swin encoder (keep in_chans=3 to match pretrained weights)
        self.encoder = SwinTransformer(
            in_chans=in_chans,
            embed_dim=128,
            window_size=[8, 8],
            patch_size=[2, 2],
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=2,
            use_v2=True,
        )
         
        if ssl_checkpoint is not None: # pretrained weights
            model_dict = torch.load(ssl_checkpoint, map_location="cpu")
            if isinstance(model_dict, dict) and "state_dict" in model_dict:
                model_dict = model_dict["state_dict"]
            # EchoCare mentions removing 'mask_token' from state_dict if present
            if isinstance(model_dict, dict) and "mask_token" in model_dict:
                model_dict.pop("mask_token")
            self.encoder.load_state_dict(model_dict, strict=False)
            print("Using pretrained self-supervised Swin backbone weights!")

        self.w_As = []  
        self.w_Bs = []

        for name, module in self.encoder.named_modules():
            if isinstance(module, WindowAttention):
                old_qkv = module.qkv
                dim = old_qkv.in_features

                # Initialize LoRA
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                module.qkv = _LoRA_qkv(
                module.qkv,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        
        self.reset_parameters()

        # Freeze original parameters; train LoRA parameters only
        for name, p in self.encoder.named_parameters():
            if "linear_a" in name or "linear_b" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # UNETR decoder blocks (same as EchoCare code)
        spatial_dims = 2
        encode_feature_size = 128
        decode_feature_size = 64
        norm_name = "instance"

        # Note: encoder1 input channels equals in_chans (=3)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=encode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * encode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * encode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * encode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # bottleneck adapt (hidden_states_out[4] channels = 16*encode_feature_size = 2048)
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * encode_feature_size,
            out_channels=16 * decode_feature_size,  # 16*64 = 1024
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5_L = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * decode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5_T = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * decode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4_L = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * decode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4_T = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * decode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3_L = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * decode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )


        self.decoder3_T = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * decode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2_L = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2_T = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1_L = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1_T = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out_L = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=seg_num_classes
        )

        self.out_T = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=seg_num_classes
        )

        # Bottleneck dimension for UniMatch head
        self.bottleneck_dim = 16 * decode_feature_size  # 1024

    def encode(self, x3):
        """
        返回 UNETR decoder 所需的 skip features + bottleneck feature：
        enc0..enc4, dec4
        """
        hidden_states_out = self.encoder(x3)  # list: [B,C,H,W]...
        enc0 = self.encoder1(x3)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        dec4 = self.encoder10(hidden_states_out[4])  # bottleneck (B,1024,h,w)
        return enc0, enc1, enc2, enc3, enc4, dec4

    def decode_long(self, enc0, enc1, enc2, enc3, enc4, dec4):
        dec3 = self.decoder5_L(dec4, enc4)
        dec2 = self.decoder4_L(dec3, enc3)
        dec1 = self.decoder3_L(dec2, enc2)
        dec0 = self.decoder2_L(dec1, enc1)
        out = self.decoder1_L(dec0, enc0)
        logits = self.out_L(out)
        return logits
    
    def decode_trans(self, enc0, enc1, enc2, enc3, enc4, dec4):
        dec3 = self.decoder5_T(dec4, enc4)
        dec2 = self.decoder4_T(dec3, enc3)
        dec1 = self.decoder3_T(dec2, enc2)
        dec0 = self.decoder2_T(dec1, enc1)
        out = self.decoder1_T(dec0, enc0)
        logits = self.out_T(out)
        return logits
    
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class Echocare_sep_dec_UniMatch(nn.Module):
    """
    Replace original UNet with SwinUNETR while preserving UniMatch training
    forward interface:
      - forward(x_long, x_trans, need_fp=False)
      - need_fp=True: returns (seg, seg_fp), (cls, cls_fp) matching UniMatch expectations
    """
    def __init__(
        self,
        in_chns: int,
        seg_class_num: int,
        cls_class_num: int,
        encoder_pth: str = "encoder.pth",
    ):
        super().__init__()

        # Ultrasound 1ch -> 3ch adapter (learnable 1x1 conv recommended over repeat)
        if in_chns == 3:
            self.in_adapter = nn.Identity()
            in_chans = 3
        else:
            self.in_adapter = nn.Conv2d(in_chns, 3, kernel_size=1, bias=False)
            in_chans = 3

        self.seg_net = SwinUNETR_Seg(
            seg_num_classes=seg_class_num,
            ssl_checkpoint=encoder_pth,
            in_chans=in_chans,
        )

        bottleneck_dim = self.seg_net.bottleneck_dim * 2  
        hidden_dim = 512  # hidden dimension (can be adjusted, e.g., 1024//2)

        # Multi-label classification head (outputs logits -> sigmoid)
        self.cls_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cls_class_num),
        )

        self.fp_dropout = nn.Dropout2d(0.5)

    def _pool_embed(self, feat_bchw: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,H,W) -> (B,C)
        return F.adaptive_avg_pool2d(feat_bchw, 1).flatten(1)

    def forward(self, x_long, x_trans, need_fp: bool = False):
        """
        Returns:
          - need_fp=False: seg_logits (B,C,H,W), cls_logits (B,K)
          - need_fp=True : ((seg, seg_fp), (cls, cls_fp)) matching UniMatch expectations
        """
        
        x = torch.cat((x_long, x_trans), dim=0)

        x3 = self.in_adapter(x)

        enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x3)

        e0_L, e0_T = enc0.chunk(2); e1_L,e1_T = enc1.chunk(2)
        e2_L, e2_T = enc2.chunk(2); e3_L, e3_T = enc3.chunk(2)
        e4_L, e4_T = enc4.chunk(2); d4_L, d4_T = dec4.chunk(2)

        if need_fp:
            def make_fp_batch(feat_L, feat_T):
                feat_all = torch.cat([feat_L, feat_T], dim=0) # [2B, C, H, W]
                perturbed = self.fp_dropout(feat_all)        # Dropout 적용
                # Long 전용 [Orig_L; FP_L], Trans 전용 [Orig_T; FP_T]로 재구성
                p_L, p_T = perturbed.chunk(2)
                return torch.cat([feat_L, p_L], dim=0), torch.cat([feat_T, p_T], dim=0)

            pe0_L, pe0_T = make_fp_batch(e0_L, e0_T)
            pe1_L, pe1_T = make_fp_batch(e1_L, e1_T)
            pe2_L, pe2_T = make_fp_batch(e2_L, e2_T)
            pe3_L, pe3_T = make_fp_batch(e3_L, e3_T)
            pe4_L, pe4_T = make_fp_batch(e4_L, e4_T)
            pd4_L, pd4_T = make_fp_batch(d4_L, d4_T)

            seg_L_combined = self.seg_net.decode_long(pe0_L, pe1_L, pe2_L, pe3_L, pe4_L, pd4_L)
            seg_T_combined = self.seg_net.decode_trans(pe0_T, pe1_T, pe2_T, pe3_T, pe4_T, pd4_T)

            seg_logits_L, seg_logits_fp_L = seg_L_combined.chunk(2)
            seg_logits_T, seg_logits_fp_T = seg_T_combined.chunk(2)

            embed_L = self._pool_embed(d4_L)
            embed_T = self._pool_embed(d4_T)
            cls_feature = torch.cat([embed_L, embed_T], dim=-1) # [B, bottleneck*2]
            cls_logits = self.cls_decoder(cls_feature)

            _, d4_L_fp = pd4_L.chunk(2)
            _, d4_T_fp = pd4_T.chunk(2)
            embed_L_fp = self._pool_embed(d4_L_fp)
            embed_T_fp = self._pool_embed(d4_T_fp)
            cls_feature_fp = torch.cat([embed_L_fp, embed_T_fp], dim=-1)
            cls_logits_fp = self.cls_decoder(cls_feature_fp)

            return (seg_logits_L, seg_logits_fp_L), (seg_logits_T, seg_logits_fp_T), (cls_logits, cls_logits_fp)

        # normal
        seg_logits_L = self.seg_net.decode_long(e0_L, e1_L, e2_L, e3_L, e4_L, d4_L)
        seg_logits_T = self.seg_net.decode_trans(e0_T, e1_T, e2_T, e3_T, e4_T, d4_T)

        # 4. 분류 헤드 (Long + Trans 특징 결합)
        embed_L = self._pool_embed(d4_L)
        embed_T = self._pool_embed(d4_T)
        cls_feature = torch.cat([embed_L, embed_T], dim=-1)
        cls_logits = self.cls_decoder(cls_feature)

        return seg_logits_L, seg_logits_T, cls_logits
