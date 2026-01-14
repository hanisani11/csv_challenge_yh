import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, num_filter, n_class, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = ConvBlock(num_filter[0] + num_filter[1], num_filter[0])
        self.conv1_1 = ConvBlock(num_filter[1] + num_filter[2], num_filter[1])
        self.conv2_1 = ConvBlock(num_filter[2] + num_filter[3], num_filter[2])
        self.conv3_1 = ConvBlock(num_filter[3] + num_filter[4], num_filter[3]) 

        self.conv0_2 = ConvBlock(num_filter[0]*2 + num_filter[1], num_filter[0])
        self.conv1_2 = ConvBlock(num_filter[1]*2 + num_filter[2], num_filter[1])
        self.conv2_2 = ConvBlock(num_filter[2]*2 + num_filter[3], num_filter[2])

        self.conv0_3 = ConvBlock(num_filter[0]*3 + num_filter[1], num_filter[0])
        self.conv1_3 = ConvBlock(num_filter[1]*3 + num_filter[2], num_filter[1])

        self.conv0_4 = ConvBlock(num_filter[0]*4 + num_filter[1], num_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(num_filter[0], n_class, kernel_size=1)
            self.final2 = nn.Conv2d(num_filter[0], n_class, kernel_size=1)
            self.final3 = nn.Conv2d(num_filter[0], n_class, kernel_size=1)
            self.final4 = nn.Conv2d(num_filter[0], n_class, kernel_size=1)
        else:
            self.final = nn.Conv2d(num_filter[0], n_class, kernel_size=1)

    def forward(self, feats):
        x0_0, x1_0, x2_0, x3_0, x4_0 = feats

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return (out1 + out2 + out3 + out4) / 4
        else:
            return self.final(x0_4)

class UNetPlusPlusTwoView(nn.Module):
    """
    Two-View UNet++:
    - Shared Encoder for Long/Trans views
    - UNet++ Dense Skip Connection Decoders
    - Combined Classification Head
    - Supports UniMatch 'need_fp' mode
    """
    def __init__(self, in_chns: int, seg_class_num: int, cls_class_num: int, deep_supervision=True):
        super().__init__()
        
        self.ft_chns = [16, 32, 64, 128, 256] 
        self.dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = ConvBlock(in_chns, self.ft_chns[0], self.dropout[0])
        self.conv1_0 = ConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.conv2_0 = ConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.conv3_0 = ConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.conv4_0 = ConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        
        self.seg_decoder_long = UNetPlusPlusDecoder(self.ft_chns, seg_class_num, deep_supervision)
        self.seg_decoder_trans = UNetPlusPlusDecoder(self.ft_chns, seg_class_num, deep_supervision)

        self.cls_fuse = nn.Sequential(
            nn.Linear(self.ft_chns[-1] * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cls_class_num)
        )

        self.apply(init_weights)

    def _shared_encoder(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return [x0_0, x1_0, x2_0, x3_0, x4_0]

    @staticmethod
    def _embed_from_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(bottleneck, 1).view(bottleneck.size(0), -1)

    def forward(self, x_long: torch.Tensor, x_trans: torch.Tensor, need_fp: bool = False):
        feat_long = self._shared_encoder(x_long)
        feat_trans = self._shared_encoder(x_trans)

        if need_fp:
            def _perturb_feats(feats):
                return [torch.cat((f, nn.Dropout2d(0.5)(f)), dim=0) for f in feats]

            p_long = _perturb_feats(feat_long)
            p_trans = _perturb_feats(feat_trans)

            seg_long = self.seg_decoder_long(p_long)
            seg_trans = self.seg_decoder_trans(p_trans)

            emb_long = self._embed_from_bottleneck(p_long[-1])
            emb_trans = self._embed_from_bottleneck(p_trans[-1])
            cls_logits = self.cls_fuse(torch.cat([emb_long, emb_trans], dim=1))

            return seg_long.chunk(2, dim=0), seg_trans.chunk(2, dim=0), cls_logits.chunk(2, dim=0)

        seg_long = self.seg_decoder_long(feat_long)
        seg_trans = self.seg_decoder_trans(feat_trans)

        emb_long = self._embed_from_bottleneck(feat_long[-1])
        emb_trans = self._embed_from_bottleneck(feat_trans[-1])
        cls_logits = self.cls_fuse(torch.cat([emb_long, emb_trans], dim=1))


        return seg_long, seg_trans, cls_logits
