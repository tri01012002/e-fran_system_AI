import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm


# Define normal convolution layer
class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 use_bn=True, use_act=True, stride=1, padding=0, dilation=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self.use_act = use_act
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if self.use_bn:
            self.batch_norm = nn.BatchNorm2d(self.out_channel)

        self.conv2d = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                kernel_size=self.kernel_size,
                                padding=self.padding, dilation=self.dilation)

    def forward(self, x):
        x = self.conv2d(x)

        if self.use_bn:
            x = self.batch_norm(x)

        if self.use_act:
            x = F.relu(x)

        return x


# Define involution
class Involution(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(Involution, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels

        self.conv1 = ConvModule(in_channel=self.channels,
                                out_channel=self.channels // self.reduction_ratio,
                                kernel_size=1)

        self.conv2 = ConvModule(in_channel=self.channels // self.reduction_ratio,
                                out_channel=self.kernel_size ** 2 * self.groups,
                                kernel_size=1,
                                stride=1,
                                use_act=False,
                                use_bn=False)
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(self.stride, self.stride)

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        # print(f"Input_shape: {x.shape}")

        weight = self.conv1(x if self.stride == 1 else self.avgpool(x))
        # print(f"Weight_shape after conv1: {weight.shape}")

        weight = self.conv2(weight)
        # print(f"Weight_shape after conv2: {weight.shape}")

        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        # print(f"Weight_shape after view: {weight.shape}")

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        # print(f"Unfold input image: {out.shape}")

        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        # print(f"Out_shape: {out.shape}")

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.involution = Involution(channels=1024, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.involution(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class GGNet(nn.Module):
    def __init__(self, num_classes):
        super(GGNet, self).__init__()

        # GoogLeNet Chosen
        base_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

        # We take all layers of GoogLeNet except the last one (Fully Connected)
        layers = list(base_model.children())[:-3]
        self.feature_extractor = nn.Sequential(*layers)

        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        # Spatial attention
        self.spatial_attention = SpatialAttention()

        # Flatten tensor
        self.flatten = nn.Flatten()

        combined_size = 14336

        # Classifiers
        self.main_classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=-1),
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, flag='train'):
        features = self.feature_extractor(x)

        # Apply spatial attention
        attention_mask = self.spatial_attention(features)
        features = features + features * attention_mask  # Residual connection

        pool1 = self.pool1(features).view(features.size(0), -1)
        pool2 = self.pool2(features).view(features.size(0), -1)
        pool3 = self.pool3(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1, pool2, pool3], dim=1)

        # Main classifier
        main_output = self.main_classifier(combined_features)

        if flag == 'train':
            aux_output = self.aux_classifier(combined_features)
            return main_output, aux_output
        if flag == 'val':
            return main_output


################################
class GGNet_P04(nn.Module):
    def __init__(self,checkpoint=None, device='cpu'):
        super(GGNet_P04, self).__init__()

        # GoogLeNet Chosen
        base_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

        # We take all layers of GoogLeNet except the last one (Fully Connected)
        layers = list(base_model.children())[:-3]
        self.feature_extractor = nn.Sequential(*layers)

        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        # Spatial attention
        self.spatial_attention = SpatialAttention()

        # Flatten tensor
        self.flatten = nn.Flatten()

        combined_size = 14336

        # Classifiers
        self.main_classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LayerNorm(normalized_shape=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LayerNorm(normalized_shape=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),
        )
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)
    def forward(self, x, flag='train'):
        features = self.feature_extractor(x)

        # Apply spatial attention
        attention_mask = self.spatial_attention(features)
        features = features + features * attention_mask  # Residual connection

        pool1 = self.pool1(features).view(features.size(0), -1)
        pool2 = self.pool2(features).view(features.size(0), -1)
        pool3 = self.pool3(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1, pool2, pool3], dim=1)

        # Main classifier
        main_output = self.main_classifier(combined_features)

        if flag == 'train':
            aux_output = self.aux_classifier(combined_features)
            return main_output, aux_output
        if flag == 'val':
            return main_output

def create_rolex_model(num_classes=3, checkpoint=None, device='cpu'):
    model = models.efficientnet_v2_s(weights=None)

    # Đóng băng tất cả parameters
    for p in model.parameters():
        p.requires_grad = False

    num_ftrs = model.classifier[1].in_features

    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3, inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(256, num_classes),
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


class MaxvitSmall(nn.Module):
    """
    MaxViT Small model for Rolex P04 authentication model 1
    """
    def __init__(self, num_classes=2, model_name='maxvit_small_tf_224.in1k',
                 pretrained=False, checkpoint=None, device='cpu'):
        super(MaxvitSmall, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)

        in_features = self.backbone.num_features

        self.backbone.reset_classifier(num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)

    def forward(self, x):
        features = self.backbone(x)  # (B, in_features)
        output = self.classifier(features)
        return output
