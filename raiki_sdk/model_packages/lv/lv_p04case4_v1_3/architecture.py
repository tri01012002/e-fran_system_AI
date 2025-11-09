import torch
import torch.nn as nn
from raiki_sdk.utils import BasicConv
from .src.models.tresnet_v2.tresnet_v2 import TResnetL_V2

class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers_FeatureHead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_FeatureHead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_FeatureHead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_FeatureHead[5])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers=None, num_classes=2):
        super().__init__()
        
        # If net_layers is None, create default TResNet backbone layers
        if net_layers is None:
            try:
                # Create TResNet backbone with proper model_params dictionary
                model_params = {'num_classes': 1000}  # Use standard ImageNet classes
                tresnet_backbone = TResnetL_V2(model_params)
                
                # Extract the feature layers from TResNet backbone
                net_layers = [
                    tresnet_backbone.body.SpaceToDepth,
                    [tresnet_backbone.body.conv1],
                    [tresnet_backbone.body.layer1],
                    [tresnet_backbone.body.layer2],
                    [tresnet_backbone.body.layer3],
                    [tresnet_backbone.body.layer4]
                ]
            except Exception as e:
                print(f"Failed to create TResNet backbone: {e}")
                # Fallback: create dummy layers structure
                net_layers = [nn.Identity() for _ in range(6)]
        
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3


def _load_model(weights_path: str, device: str) -> Network_Wrapper:
    # Load full model object
    import __main__
    __main__.Network_Wrapper = Network_Wrapper
    __main__.Features = Features
    # Also register the old import paths that might be in the saved model
    import sys
    from types import ModuleType
    
    # Create a fake 'src' module in sys.modules to handle old import paths
    if 'src' not in sys.modules:
        src_module = ModuleType('src')
        sys.modules['src'] = src_module
        
        # Create the nested module structure
        src_models = ModuleType('src.models')
        sys.modules['src.models'] = src_models
        src_module.models = src_models
        
        src_tresnet = ModuleType('src.models.tresnet')
        sys.modules['src.models.tresnet'] = src_tresnet
        src_models.tresnet = src_tresnet
        
        src_tresnet_v2 = ModuleType('src.models.tresnet_v2')
        sys.modules['src.models.tresnet_v2'] = src_tresnet_v2
        src_models.tresnet_v2 = src_tresnet_v2
        
        src_tresnet_layers = ModuleType('src.models.tresnet.layers')
        sys.modules['src.models.tresnet.layers'] = src_tresnet_layers
        src_tresnet.layers = src_tresnet_layers
        
        src_tresnet_v2_layers = ModuleType('src.models.tresnet_v2.layers')
        sys.modules['src.models.tresnet_v2.layers'] = src_tresnet_v2_layers
        src_tresnet_v2.layers = src_tresnet_v2_layers
        
        # Import the actual modules and assign them to the fake namespace
        from .src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer, DownsampleJIT, Downsample
        from .src.models.tresnet.layers.avg_pool import FastGlobalAvgPool2d
        from .src.models.tresnet.layers.space_to_depth import SpaceToDepthModule, SpaceToDepthJit, SpaceToDepth
        from .src.models.tresnet_v2.tresnet_v2 import TResnetL_V2, BasicBlock, Bottleneck, TResNetV2
        from .src.models.tresnet_v2.layers.squeeze_and_excite import SEModule as SEModuleV2
        from raiki_sdk.utils import BasicConv
        
        # Set up tresnet namespace
        src_tresnet_layers.AntiAliasDownsampleLayer = AntiAliasDownsampleLayer
        src_tresnet_layers.avg_pool = ModuleType('src.models.tresnet.layers.avg_pool')
        src_tresnet_layers.avg_pool.FastGlobalAvgPool2d = FastGlobalAvgPool2d
        src_tresnet_layers.space_to_depth = ModuleType('src.models.tresnet.layers.space_to_depth')
        src_tresnet_layers.space_to_depth.SpaceToDepthModule = SpaceToDepthModule
        src_tresnet_layers.space_to_depth.SpaceToDepthJit = SpaceToDepthJit
        src_tresnet_layers.space_to_depth.SpaceToDepth = SpaceToDepth
        
        # Set up tresnet_v2 namespace
        src_tresnet_v2.tresnet_v2 = ModuleType('src.models.tresnet_v2.tresnet_v2')
        src_tresnet_v2.tresnet_v2.TResnetL_V2 = TResnetL_V2
        src_tresnet_v2.tresnet_v2.BasicBlock = BasicBlock
        src_tresnet_v2.tresnet_v2.Bottleneck = Bottleneck
        src_tresnet_v2.tresnet_v2.TResNetV2 = TResNetV2
        src_tresnet_v2_layers.squeeze_and_excite = ModuleType('src.models.tresnet_v2.layers.squeeze_and_excite')
        src_tresnet_v2_layers.squeeze_and_excite.SEModule = SEModuleV2
        
        # Register all modules in sys.modules
        sys.modules['src.models.tresnet.layers.anti_aliasing'] = ModuleType('src.models.tresnet.layers.anti_aliasing')
        sys.modules['src.models.tresnet.layers.anti_aliasing'].AntiAliasDownsampleLayer = AntiAliasDownsampleLayer
        sys.modules['src.models.tresnet.layers.anti_aliasing'].DownsampleJIT = DownsampleJIT
        sys.modules['src.models.tresnet.layers.anti_aliasing'].Downsample = Downsample
        sys.modules['src.models.tresnet.layers.avg_pool'] = src_tresnet_layers.avg_pool
        sys.modules['src.models.tresnet.layers.space_to_depth'] = src_tresnet_layers.space_to_depth
        sys.modules['src.models.tresnet_v2.tresnet_v2'] = src_tresnet_v2.tresnet_v2
        sys.modules['src.models.tresnet_v2.layers.squeeze_and_excite'] = src_tresnet_v2_layers.squeeze_and_excite
        
        # Handle basic_conv import
        basic_conv_module = ModuleType('basic_conv')
        basic_conv_module.BasicConv = BasicConv
        sys.modules['basic_conv'] = basic_conv_module
        
    loaded_model = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(loaded_model, dict):
        # It's a state_dict
        model_auth_LVP04 = Network_Wrapper(num_classes=2)
        model_auth_LVP04.load_state_dict(loaded_model)
    else:
        print("It's a full model object")
        # It's a full model object
        model_auth_LVP04 = loaded_model

    model_auth_LVP04.to(device)
    model_auth_LVP04.eval()

    return loaded_model