import math
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── FastGlobalAvgPool2d ───────────────────────────────────────────────────────
class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


# ── SEModule ──────────────────────────────────────────────────────────────────
class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels, inplace=True):
        super().__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc2(self.relu(self.fc1(x_se)))
        return x * self.activation(x_se)


# ── SpaceToDepth / SpaceToDepthModule ─────────────────────────────────────────
class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):  # param kept for API compat
        super().__init__()
        self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


# ── AntiAliasDownsampleLayer ──────────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super().__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        input_pad = F.pad(x, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3,
                 stride: int = 2, channels: int = 0):
        super().__init__()
        self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


# ── InPlaceABN shim (fallback nếu không cài inplace_abn) ─────────────────────
try:
    from inplace_abn import InPlaceABN, ABN
    _HAVE_IABN = True
except ImportError:
    _HAVE_IABN = False

    class _ABNBase(nn.Module):
        def __init__(self, num_features, activation='leaky_relu', activation_param=1e-2):
            super().__init__()
            self.num_features = num_features
            self.activation = activation
            self.activation_param = activation_param
            self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
            if activation == 'leaky_relu':
                self.act = nn.LeakyReLU(negative_slope=activation_param, inplace=True)
            elif activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.bn(x))

        @property
        def weight(self): return self.bn.weight
        @property
        def bias(self): return self.bn.bias
        @property
        def eps(self): return self.bn.eps

    class InPlaceABN(_ABNBase): pass
    class ABN(_ABNBase): pass


def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    if isinstance(module, InPlaceABN):
        module_new = ABN(module.num_features, activation=module.activation,
                         activation_param=module.activation_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        if _HAVE_IABN:
            module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3,
               activation_param=1e-2, groups=1):
    activation_param = 1e-6
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, groups=groups, bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


# ── BasicBlock ────────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 use_se=True, anti_alias_layer=None):
        super().__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                    anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        return self.relu(out)


# ── Bottleneck ────────────────────────────────────────────────────────────────
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 use_se=True, anti_alias_layer=None):
        super().__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1,
                                activation="leaky_relu", activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                    activation="leaky_relu", activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2,
                                        activation="leaky_relu", activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                               activation="leaky_relu", activation_param=1e-3),
                    anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1,
                                stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + residual
        return self.relu(out)


# ── TResNetV2 ─────────────────────────────────────────────────────────────────
class TResNetV2(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000,
                 width_factor=1.0, remove_model_jit=False):
        super().__init__()
        self.inplanes = int(int(64 * width_factor + 4) / 8) * 8
        self.planes   = int(int(64 * width_factor + 4) / 8) * 8

        SpaceToDepth_m    = SpaceToDepthModule(remove_model_jit=True)
        conv1             = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        anti_alias_layer  = partial(AntiAliasDownsampleLayer, remove_aa_jit=True)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        layer1 = self._make_layer(Bottleneck, self.planes,     layers[0], stride=1,
                                  use_se=True,  anti_alias_layer=anti_alias_layer)
        layer2 = self._make_layer(Bottleneck, self.planes * 2, layers[1], stride=2,
                                  use_se=True,  anti_alias_layer=anti_alias_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2,
                                  use_se=True,  anti_alias_layer=anti_alias_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2,
                                  use_se=False, anti_alias_layer=anti_alias_layer)

        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', SpaceToDepth_m),
            ('conv1',  conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4),
        ]))

        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.global_pool  = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.head         = nn.Sequential(OrderedDict([('fc', nn.Linear(self.num_features, num_classes))]))
        self.embeddings   = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, InPlaceABN)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init last BN/ABN weight in residual blocks
        for m in self.modules():
            if isinstance(m, BasicBlock):
                abn = m.conv2[1]
                if 'weight' in abn._parameters:
                    abn._parameters['weight'] = nn.Parameter(torch.zeros_like(abn.weight))
                elif hasattr(abn, 'bn'):
                    nn.init.constant_(abn.bn.weight, 0)
            if isinstance(m, Bottleneck):
                abn = m.conv3[1]
                if 'weight' in abn._parameters:
                    abn._parameters['weight'] = nn.Parameter(torch.zeros_like(abn.weight))
                elif hasattr(abn, 'bn'):
                    nn.init.constant_(abn.bn.weight, 0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2,
                                           ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=1, activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = [block(self.inplanes, planes, stride, downsample,
                        use_se=use_se, anti_alias_layer=anti_alias_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return self.head(self.global_pool(x))


def TResnetL_V2(model_params):
    """Constructs a large TResNetV2 model (L variant)."""
    return TResNetV2(
        layers=[3, 4, 23, 3],
        num_classes=model_params['num_classes'],
        in_chans=3,
        width_factor=1.0,
        remove_model_jit=True,
    )


# ── BasicConv (dùng trong CMAL head) ─────────────────────────────────────────
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn  = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.act = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn  is not None: x = self.bn(x)
        if self.act is not None: x = self.act(x)
        return x


# ── Features (backbone wrapper) ───────────────────────────────────────────────
class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super().__init__()
        self.net_layer_0 = nn.Sequential(net_layers_FeatureHead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_FeatureHead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_FeatureHead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_FeatureHead[5])

    def forward(self, x):
        x  = self.net_layer_0(x)
        x  = self.net_layer_1(x)
        x  = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)
        return x1, x2, x3


# ── Network_Wrapper (CMAL multi-head) ─────────────────────────────────────────
class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_classes):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512,  512,  kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512,  1024, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512,  kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512,  1024, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512,  kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512,  1024, kernel_size=3, stride=1, padding=1, relu=True),
        )

        def _head():
            return nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ELU(inplace=True),
                nn.Linear(512, num_classes),
            )

        self.classifier1       = _head()
        self.classifier2       = _head()
        self.classifier3       = _head()
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_  = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_  = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        x2_  = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_  = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_  = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_  = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = self.classifier_concat(torch.cat((x1_f, x2_f, x3_f), dim=-1))

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3