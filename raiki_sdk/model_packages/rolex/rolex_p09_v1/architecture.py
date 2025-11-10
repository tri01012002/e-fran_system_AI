import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision
import torch.nn.functional as F


##########################

class ReducedMultiTaskNetwork(nn.Module):

    def __init__(self, num_class):
        super(ReducedMultiTaskNetwork, self).__init__()
        base_model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        layers = list(base_model.children())[:-2]
        self.feature_extractor = torch.nn.Sequential(*layers)

        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        # Main classifier only
        self.main_classifier = nn.Sequential(
            nn.Linear(28672, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Need to modify for each part
            nn.Linear(512, num_class)
        )

        # Placeholder for gradients and activations
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, flag="val"):
        features = self.feature_extractor(x)
        if flag == "heatmap":
            # Register hook to capture gradients and save activations
            features.register_hook(self.save_gradient)
            self.activations = features

        pool1 = self.pool1(features).view(features.size(0), -1)
        pool2 = self.pool2(features).view(features.size(0), -1)
        pool3 = self.pool3(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1, pool2, pool3], dim=1)
        main_output = self.main_classifier(combined_features)
        return main_output


############################


########################################

# Define Convolution
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
        weight = self.conv1(x if self.stride == 1 else self.avgpool(x))
        weight = self.conv2(weight)

        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)

        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)

        return out


# Define Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Changed the channel size to 2048 to match ResNet50's final feature map
        self.involution = Involution(channels=2048, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.involution(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ReducedMultiTaskNetwork_Involution(nn.Module):
    def __init__(self, num_classes, checkpoint = None, device = 'cpu'):
        super(ReducedMultiTaskNetwork_Involution, self).__init__()
        base_model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        layers = list(base_model.children())[:-2]
        self.feature_extractor = torch.nn.Sequential(*layers)

        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        # Spatial attention with corrected channel size
        self.spatial_attention = SpatialAttention()

        # Changed total_features calculation to use 2048 (ResNet50's final feature channels)
        total_features = 2048 + 8192 + 18432

        # Main classifier only
        self.main_classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict, strict=False)
        # Placeholder for gradients and activations
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def get_activations(self, images):
        return self.conv(images)

    def forward(self, x, flag="train"):
        features = self.feature_extractor(x)
        if flag == "heatmap":
            # Register hook to capture gradients and save activations
            features.register_hook(self.save_gradient)
            self.activations = features

        attention_mask = self.spatial_attention(features)
        features = features + features * attention_mask  # Added missing attention application

        pool1 = self.pool1(features).view(features.size(0), -1)
        pool2 = self.pool2(features).view(features.size(0), -1)
        pool3 = self.pool3(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1, pool2, pool3], dim=1)
        main_output = self.main_classifier(combined_features)
        return main_output


###############################################
import timm


class ReducedMultiTaskNetwork_P03(nn.Module):

    def __init__(self, num_class):
        super(ReducedMultiTaskNetwork_P03, self).__init__()
        base_model = timm.create_model(
            'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k',
            pretrained=False,
            features_only=True,
        )

        layers = list(base_model.children())
        layers[0].conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.feature_extractor = torch.nn.Sequential(*layers)

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))

        # Main classifier only
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),  # Normalize pooled features
            nn.Linear(768, 512),  # MLP: First linear layer
            nn.GELU(),  # GELU activation function
            nn.Dropout(p=0.25),  # Dropout for regularization
            nn.Linear(512, num_class)  # Final output layer (classification)
        )
        # Placeholder for gradients and activations
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, flag="val"):
        features = self.feature_extractor(x)
        features = features.permute(0, 3, 1, 2).contiguous()
        if flag == "heatmap":
            # Register hook to capture gradients and save activations
            features.register_hook(self.save_gradient)
            self.activations = features

        pool1 = self.pool1(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1], dim=1)

        main_output = self.main_classifier(combined_features)
        return main_output


######################################
class ReducedMultiTaskNetworkMamba(nn.Module):
    def __init__(self, num_classes):
        super(ReducedMultiTaskNetworkMamba, self).__init__()
        base_model = timm.create_model(
            'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k',
            pretrained=False,
            features_only=True,
        )

        layers = list(base_model.children())

        self.feature_extractor = torch.nn.Sequential(*layers)

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))

        # Main classifier only
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),  # Normalize pooled features
            nn.Linear(768, 512),  # MLP: First linear layer
            nn.GELU(),  # GELU activation function
            nn.Dropout(p=0.25),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Final output layer (classification)
        )
        # Placeholder for gradients and activations
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, flag="val"):
        features = self.feature_extractor(x)
        features = features.permute(0, 3, 1, 2).contiguous()
        if flag == "heatmap":
            # Register hook to capture gradients and save activations
            features.register_hook(self.save_gradient)
            self.activations = features

        pool1 = self.pool1(features).view(features.size(0), -1)

        combined_features = torch.cat([pool1], dim=1)

        main_output = self.main_classifier(combined_features)
        return main_output
