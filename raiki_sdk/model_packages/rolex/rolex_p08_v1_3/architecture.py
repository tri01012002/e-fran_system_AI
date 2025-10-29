import torch
import timm
import torch.nn as nn

class ReducedNetworkCaformer(nn.Module):
    def __init__(self, num_classes, model_name='caformer_m36.sail_in22k_ft_in1k',
                 dropout_main=0.25, freeze_backbone=False,
                 checkpoint=None, device='cpu'):
        super(ReducedNetworkCaformer, self).__init__()

        self.backbone_name = model_name
        print(f"Khởi tạo ReducedNetworkCaformer với backbone: {self.backbone_name}")

        # ✅ Load model
        try:
            self.base_model = timm.create_model(
                self.backbone_name, pretrained=False, features_only=True, in_chans=3
            )
        except Exception as e:
            print(f"LỖI: Không thể tạo model '{self.backbone_name}'. Vui lòng kiểm tra lại tên model.")
            print(e)
            raise e

        # ✅ Auto num_features
        try:
            self.num_features = self.base_model.feature_info.channels(-1)
            print(f"Tự động xác định số features output: {self.num_features}")
        except AttributeError:
            warnings.warn(f"Không tìm thấy 'feature_info' cho {self.backbone_name}. Cần kiểm tra num_features thủ công!")
            try:
                self.num_features = self.base_model.num_features
                warnings.warn(f"Lấy num_features từ thuộc tính '.num_features': {self.num_features}. HÃY KIỂM TRA LẠI!")
            except AttributeError:
                raise ValueError(f"Không thể tự động xác định num_features cho {self.backbone_name}.")

        # ✅ Freeze backbone if needed
        for param in self.base_model.parameters():
            param.requires_grad = not freeze_backbone

        # ✅ Classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(self.num_features, eps=1e-6, elementwise_affine=True),
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_main),
            nn.Linear(512, num_classes)
        )

        # ✅ Load checkpoint if provided
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict, strict=False)

        # ✅ Gradient & Activation holder
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, flag="val"):
        features_list = self.base_model(x)
        features = features_list[-1] if isinstance(features_list, list) else features_list

        self.activations = features
        if features.requires_grad and flag == "heatmap":
            features.register_hook(self.save_gradient)

        pooled_features = self.pool(features).flatten(start_dim=1)
        return self.main_classifier(pooled_features)

    def get_activations(self):
        return self.activations

    def get_gradients(self):
        return self.gradients

