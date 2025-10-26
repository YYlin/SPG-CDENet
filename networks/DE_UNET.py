import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class ResNet50_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        # 全部采用 bilinear 上采样
        self.upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
            nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)
        )

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x, ippm_x):
        if ippm_x is not None:
            up_x = up_x + ippm_x

        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


def RF_Block(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=False))


# 以下是cross-attention的代码
class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, o):
        fused = torch.cat([x, o], dim=1)
        return self.conv(fused)


class CrossAttention(nn.Module):
    """Cross Attention 融合"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn_x_to_o = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_o_to_x = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x, o):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        o_flat = o.flatten(2).transpose(1, 2)

        x2o, _ = self.attn_x_to_o(x_flat, o_flat, o_flat)
        o2x, _ = self.attn_o_to_x(o_flat, x_flat, x_flat)

        x_out = x_flat + o2x
        o_out = o_flat + x2o

        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        o_out = o_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out, o_out


class DENet_Cross_Dual_Encoder(nn.Module):
    def __init__(self, num_classes=9, DEPTH=6, fusion_type="sum"):
        """
        fusion_type: "none" / "sum" / "concat" / "crossattn"
        """
        super().__init__()
        self.DEPTH = DEPTH
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks, down_blocks_organ = [], []
        up_blocks, ippm_blocks = [], []

        # 双输入编码器
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.input_block_organ = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool_organ = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks_organ.append(bottleneck)
        self.down_blocks_organ = nn.ModuleList(down_blocks_organ)

        # 设置不同融合模块
        if fusion_type == "concat":
            self.fusions = nn.ModuleList([
                FusionConv(256, 256),
                FusionConv(512, 512),
                FusionConv(1024, 1024),
                FusionConv(2048, 2048)
            ])
        elif fusion_type == "crossattn":
            self.fusions = nn.ModuleList([
                CrossAttention(256),
                CrossAttention(512),
                CrossAttention(1024),
                CrossAttention(2048)
            ])

        # bridge
        self.bridge = Bridge(4096, 2048)
        ippm_blocks.append(RF_Block(2048, 1024, 2))
        ippm_blocks.append(RF_Block(2048, 512, 4))
        ippm_blocks.append(RF_Block(2048, 256, 8))
        ippm_blocks.append(RF_Block(2048, 128, 16))
        self.ippm_blocks = nn.ModuleList(ippm_blocks)

        up_blocks.append(ResNet50_UpBlock(2048, 1024))
        up_blocks.append(ResNet50_UpBlock(1024, 512))
        up_blocks.append(ResNet50_UpBlock(512, 256))
        up_blocks.append(ResNet50_UpBlock(128 + 64, 128, 256, 128))
        up_blocks.append(ResNet50_UpBlock(64 + 3, 64, 128, 64))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)

    def forward(self, x, organ):
        if x.size()[1] == 1 or organ.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            organ = organ.repeat(1, 3, 1, 1)

        pre_pools, pre_pools_organ = {}, {}
        pre_pools["layer_0"] = x
        pre_pools_organ["layer_0"] = organ

        x = self.input_block(x)
        organ = self.input_block_organ(organ)

        pre_pools["layer_1"] = x
        pre_pools_organ["layer_1"] = organ

        x = self.input_pool(x)
        organ = self.input_pool_organ(organ)

        for i, (block_x, block_o) in enumerate(zip(self.down_blocks, self.down_blocks_organ), 2):
            x = block_x(x)
            organ = block_o(organ)

            # ================= 融合 =================
            if self.fusion_type == "sum":
                fused = x + organ
                x, organ = fused, fused
            elif self.fusion_type == "concat":
                fused = self.fusions[i-2](x, organ)
                x, organ = fused, fused
            elif self.fusion_type == "crossattn":
                x, organ = self.fusions[i-2](x, organ)

            if i == (self.DEPTH - 1):
                continue

            pre_pools[f"layer_{i}"] = x
            pre_pools_organ[f"layer_{i}"] = organ

        # bridge
        x = torch.cat([organ, x], dim=1)
        x = self.bridge(x)

        # decoder
        IPPM_List = [None] + [block(x) for block in self.ippm_blocks]
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key], IPPM_List[i - 1])

        return self.out(x)

# fusion_type: "none" / "sum" / "concat" / "crossattn"
# model = DENet_Cross_Dual_Encoder(fusion_type='crossattn')
# inp = torch.rand((2, 3, 224, 224))
# out = model(inp, inp)
# print(out.shape)
