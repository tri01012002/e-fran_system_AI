import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import math
from typing import Optional

# ==================== ArcFace Loss ====================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


# ==================== MViT + ArcFace Model ====================
class MViTArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, pretrained=True, 
                 model_name='maxvit_small_tf_384.in1k', s=30.0, m=0.50, img_size=224):
        super(MViTArcFaceModel, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            backbone_features = self.backbone(dummy_input).shape[1]
        
        self.embedding = nn.Linear(backbone_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcFaceLoss(embedding_size, num_classes, s=s, m=m)
        
    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.bn(self.embedding(features))
        
        if labels is None:
            return F.normalize(embeddings, p=2, dim=1)
        
        return self.arcface(embeddings, labels)




    