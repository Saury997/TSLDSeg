import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Downsample(nn.Module):
    """Resolution down-sampling layer"""

    def __init__(self, num_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(num_channels,
                                        num_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class HSEM(nn.Module):
    """Hyper Semantic Condition Extraction Module"""
    def __init__(self, in_channels, out_channels, dist_type='p-norm'):
        super().__init__()
        self.dist_type = dist_type
        self.threshold = 10 if dist_type == 'p-norm' else 0.2
        self.hgconv = HyPConv(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.down = Downsample(in_channels)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()   # B, H*W, C
        feature = x.clone()
        distance = torch.cdist(feature, feature) if self.dist_type == 'p-norm' else self.cosine_distance(feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))
        x = self.down(x)
        x = self.out_conv(x)

        return x

    @staticmethod
    def cosine_distance(pixel_features):
        """
        计算像素之间的余弦距离。
        args: pixel_features: Tensor 形状为 [B, H*W, C]，表示像素特征
        return:cosine_dist: Tensor 形状为 [B, H*W, H*W]，表示像素两两之间的余弦距离
        """
        # 归一化特征，使其模长为1
        normalized_features = F.normalize(pixel_features, p=2, dim=-1)

        # 计算余弦相似度 [B, H*W, H*W]
        cosine_sim = torch.bmm(normalized_features, normalized_features.transpose(1, 2))

        # 余弦距离 = 1 - 余弦相似度
        cosine_dist = 1 - cosine_sim

        return cosine_dist


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)
        return x


class EdgeConv2d(nn.Module):
    def __init__(self, kernel_type, in_channels, out_channels):
        super(EdgeConv2d, self).__init__()
        self.type = kernel_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels
        conv0 = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0)
        self.k0 = conv0.weight
        self.b0 = conv0.bias

        scale = torch.randn((self.out_channels, 1, 1, 1)) * 1e-3
        self.scale = nn.Parameter(scale)
        bias = torch.randn(self.out_channels) * 1e-3
        self.bias = nn.Parameter(bias)

        if self.type == 'conv1x1-conv3x3':
            conv1 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            mask = self.get_mask(self.out_channels, kernel_type='sobelx')
            self.mask = nn.Parameter(mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            mask = self.get_mask(self.out_channels, kernel_type='sobely')
            self.mask = nn.Parameter(mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            mask = self.get_mask(self.out_channels, kernel_type='laplacian')
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            raise ValueError('Unsupported seq_type!')

    def get_mask(self, num_channels, kernel_type):
        mask = torch.zeros((num_channels, 1, 3, 3), dtype=torch.float32)
        if kernel_type == 'laplacian':
            for i in range(num_channels):
                mask[i, 0, 0, 1] = 1.0  # 上方中间
                mask[i, 0, 1, 0] = 1.0  # 左侧中间
                mask[i, 0, 1, 1] = -4.0  # 中心
                mask[i, 0, 1, 2] = 1.0  # 右侧中间
                mask[i, 0, 2, 1] = 1.0  # 下方中间
        elif kernel_type == 'sobelx':
            for i in range(self.out_channels):
                mask[i, 0, 0, 0] = -1.0
                mask[i, 0, 0, 1] = -2.0
                mask[i, 0, 0, 2] = -1.0
                mask[i, 0, 2, 0] = 1.0
                mask[i, 0, 2, 1] = 2.0
                mask[i, 0, 2, 2] = 1.0
        elif kernel_type == 'sobely':
            for i in range(self.out_channels):
                mask[i, 0, 0, 0] = -1.0
                mask[i, 0, 0, 2] = 1.0
                mask[i, 0, 1, 0] = -2.0
                mask[i, 0, 1, 2] = 2.0
                mask[i, 0, 2, 0] = -1.0
                mask[i, 0, 2, 2] = 1.0
        else:
            raise ValueError('Unsupported seq_type!')
        return mask

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            y1 = F.conv2d(y0, self.k1, self.b1, stride=1)
        else:
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            y1 = F.conv2d(y0, self.scale * self.mask, self.bias, stride=1, groups=self.out_channels)
        return y1

class EEB(nn.Module):
    """Edge Extraction Block"""
    def __init__(self, in_channels, out_channels):
        super(EEB, self).__init__()
        self.sobelx_conv = EdgeConv2d(kernel_type='conv1x1-sobelx', in_channels=in_channels, out_channels=out_channels)
        self.sobely_conv = EdgeConv2d(kernel_type='conv1x1-sobely', in_channels=in_channels, out_channels=out_channels)
        self.laplacian_conv = EdgeConv2d(kernel_type='conv1x1-laplacian', in_channels=in_channels, out_channels=out_channels)
        self.conv3x3 = EdgeConv2d(kernel_type='conv1x1-conv3x3', in_channels=in_channels, out_channels=out_channels)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        sobelx = self.sobelx_conv(x)
        sobely = self.sobely_conv(x)
        laplacian = self.laplacian_conv(x)
        conv3x3 = self.conv3x3(x)
        edge_feat = sobelx + sobely + laplacian + conv3x3
        out = self.norm(edge_feat)

        return out


class MSOEB(nn.Module):
    """Multi-scale Orientations Extraction Block"""
    def __init__(self, in_channels, out_channels):
        super(MSOEB, self).__init__()
        self.stem = nn.Sequential(
            nn.BatchNorm2d(in_channels), # InstanceNorm2d
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
            nn.LeakyReLU()
        )

        self.scale_1 = nn.Sequential(
            # GaborConv2d(out_channels, out_channels, kernel_size=(1,1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU()
        )
        self.scale_2 = nn.Sequential(
            # GaborConv2d(out_channels, out_channels, kernel_size=(3,3)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3 // 2, groups=out_channels),
            nn.LeakyReLU()
        )
        self.scale_3 = nn.Sequential(
            # GaborConv2d(out_channels, out_channels, kernel_size=(5,5)),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=5 // 2, groups=out_channels),
            nn.LeakyReLU()
        )

        self.sigma0 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.sigma1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.sigma2 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        s = self.stem(x)  # (B,C,H,W)
        s1 = self.scale_1(s)
        s2 = self.scale_2(s)
        s3 = self.scale_3(s)
        y = self.sigma0 * s1 + self.sigma1 * s2 + self.sigma2 * s3  # (B,C,H,W)
        return self.out_conv(y)

class OEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OEB, self).__init__()
        self.ms_blk = MSOEB(in_channels, out_channels)
        self.gabor_conv = GaborConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(7, 7))
        self.out_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        ms_feat = self.ms_blk(x)
        gabor_feat = self.gabor_conv(ms_feat)
        out = self.out_conv(gabor_feat)
        return out


class TAM(nn.Module):
    """Texture-aware Module"""
    def __init__(self, in_channels, out_channels):
        super(TAM, self).__init__()
        self.orient_block = OEB(in_channels=in_channels, out_channels=in_channels // 2)
        self.edge_block = EEB(in_channels=in_channels, out_channels=in_channels // 2)
        self.down = Downsample(num_channels=in_channels)
        self.out_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        orient_feat = self.orient_block(x)
        edge_feat = self.edge_block(x)
        texture_feat = torch.cat([orient_feat, edge_feat], dim=1)
        out = self.down(texture_feat)
        return self.out_conv(out)


class GaborConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
    ):
        super().__init__()
        padding = kernel_size[0] // 2

        self.conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size  # (h, w)
        self.delta = 1e-3

        pi = torch.tensor(math.pi)
        n_rand = torch.randint(0, 5, (out_channels, in_channels), dtype=torch.float32)
        self.freq = nn.Parameter((pi / 2) * (torch.sqrt(torch.tensor(2.0))) ** (-n_rand), requires_grad=True)

        theta_rand = torch.randint(0, 8, (out_channels, in_channels), dtype=torch.float32)
        self.theta = nn.Parameter((pi / 8) * theta_rand, requires_grad=True)

        self.sigma = nn.Parameter(pi / self.freq, requires_grad=True)
        self.psi = nn.Parameter(pi * torch.rand(out_channels, in_channels), requires_grad=True)

        h, w = self.kernel_size
        self.x0 = nn.Parameter(
            torch.ceil(torch.Tensor([w / 2.0]))[0], requires_grad=False
        )
        self.y0 = nn.Parameter(
            torch.ceil(torch.Tensor([h / 2.0]))[0], requires_grad=False
        )
        x_lin = torch.linspace(-self.x0 + 1, self.x0, steps=w)
        y_lin = torch.linspace(-self.y0 + 1, self.y0, steps=h)
        y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing='ij')
        self.register_buffer('x_grid', x_grid.clone())
        self.register_buffer('y_grid', y_grid.clone())

        self.is_calculated = False

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        else:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    @torch.no_grad()
    def calculate_weights(self):
        O, I = self.freq.shape
        h, w = self.kernel_size
        x = self.x_grid.unsqueeze(0).unsqueeze(0)  # shape (1, 1, h, w)
        y = self.y_grid.unsqueeze(0).unsqueeze(0)  # shape (1, 1, h, w)
        freq = self.freq.unsqueeze(-1).unsqueeze(-1)  # shape (O, I, 1, 1)
        theta = self.theta.unsqueeze(-1).unsqueeze(-1)  # shape (O, I, 1, 1)
        sigma = self.sigma.unsqueeze(-1).unsqueeze(-1)  # shape (O, I, 1, 1)
        psi = self.psi.unsqueeze(-1).unsqueeze(-1)  # shape (O, I, 1, 1)

        rotx = x * torch.cos(theta) + y * torch.sin(theta)
        roty = -x * torch.sin(theta) + y * torch.cos(theta)

        exponent = -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
        g_exp = torch.exp(exponent)
        g_cos = torch.cos(freq * rotx + psi)
        factor = (freq ** 2) / (2 * math.pi * (sigma ** 2))
        gabor = factor * g_exp * g_cos  # shape (O, I, h, w)
        self.conv_layer.weight.copy_(gabor)

    def _forward_unimplemented(self, *inputs):
        raise NotImplementedError("This method is not implemented.")


if __name__ == '__main__':
    x = torch.randn((2, 256, 64, 64))
    x = x.cuda()
    model = TAM(in_channels=256, out_channels=8)
    model.cuda()
    out = model(x)
    print(out.shape)

    model = HSEM(in_channels=256, out_channels=8, dist_type='cos').cuda()
    out = model(x)
    print(out.shape)
    layer = GaborConv2d(in_channels=3, out_channels=8, kernel_size=(7, 7))
    x = torch.randn(4, 3, 32, 32)
    y = layer(x)
    print(y.shape)
