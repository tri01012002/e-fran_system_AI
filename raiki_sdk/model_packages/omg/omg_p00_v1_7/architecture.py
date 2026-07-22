import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from PIL import Image
from ._load_model import _load_database

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
                 model_name='vit_base_patch16_dinov3.lvd1689m', s=30.0, m=0.50, img_size=224):
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


# ==================== Rolex P00 Classifier Model ====================
class FaceRecognizerBagLV:
    """Singleton Face Recognition class with mechanism to save multiple vectors / person and Top-K Recognition"""

    _instance = None  # Keep only one instance
    _embeddings_db = None  # Cache embeddings to avoid reloading

    def __new__(cls, model=None, transform=None, vector_db_file = None, device='cuda'):
        # If instance already exists, return it (skip new initialization)
        if cls._instance is not None:
            return cls._instance

        # If not, create new instance
        instance = super(FaceRecognizerBagLV, cls).__new__(cls)
        cls._instance = instance  # Save singleton
        return instance

    def __init__(self, model=None, transform=None, vector_db_file=None,  device='cuda'):
        # Only init once (avoid re-init if called again)
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model = model.to(device)
        self.model.eval()
        self.transform = transform
        self.device = device

        # Load embeddings only once
        if FaceRecognizerBagLV._embeddings_db is None:
            FaceRecognizerBagLV._embeddings_db = _load_database(vector_db_file,device)
        self.embeddings_db = FaceRecognizerBagLV._embeddings_db
        self.idx_to_class = {}

        self._initialized = True  # Mark as initialized

    # ---------------- EMBEDDING EXTRACTION ----------------
    def extract_embedding_from_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """Extract embedding from PIL image"""
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_tensor)
        return emb.squeeze(0)

    # ---------------- COSINE SIMILARITY ----------------
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors"""
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        return torch.dot(emb1, emb2).item()

    # ---------------- RECOGNITION ----------------
    def recognize_pil(self, pil_image: Image.Image, top_k: int = 3, N_closest: int = 3, return_all_scores: bool = False):
        """Recognize face from PIL image"""
        query_embedding = self.extract_embedding_from_pil(pil_image)
        query_embedding = F.normalize(query_embedding, p=2, dim=0)

        person_scores = {}
        for name, emb_list in self.embeddings_db.items():
            sims = [self.cosine_similarity(query_embedding, e) for e in emb_list]
            sims = sorted(sims, reverse=True)
            top_n = sims[:min(len(sims), N_closest)]
            avg_sim = sum(top_n) / len(top_n)
            person_scores[name] = avg_sim

        sorted_results = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)
        if return_all_scores:
            return sorted_results
        return sorted_results[:top_k]


    def recognize_pil_double(
            self,
            pil_image: Image.Image,
            top_k: int = 3,
        N_closest: int = 3
        ):
        """Query original image + grayscale image but filter by key type before merge"""

        # --- 1. Embedding original image ---
        q_color = self.extract_embedding_from_pil(pil_image)
        q_color = F.normalize(q_color, p=2, dim=0)

        # --- 2. Convert to grayscale ---
        gray_image =  pil_image.convert("L").convert("RGB")

        # --- 3. Embedding grayscale image ---
        q_gray = self.extract_embedding_from_pil(gray_image)
        q_gray = F.normalize(q_gray, p=2, dim=0)

        # Function to calculate score with name filtering
        def compute_scores(query_embedding, only_gray=False):
            scores = {}
            for name, emb_list in self.embeddings_db.items():

                # Filter by name
                if only_gray:
                    if "_gray" not in name:
                        continue
                else:
                    if "_gray" in name:
                        continue

                sims = [self.cosine_similarity(query_embedding, e) for e in emb_list]
                sims = sorted(sims, reverse=True)

                top_n = sims[:min(len(sims), N_closest)]
                avg_sim = sum(top_n) / len(top_n)

                scores[name] = avg_sim
            return scores

        # --- 4. Calculate score ---
        color_scores = compute_scores(q_color, only_gray=False)  # only name
        gray_scores  = compute_scores(q_gray,  only_gray=True)   # only name_gray

        # --- 5. Merge by base_name ---
        merged = {}

        for name, score in color_scores.items():
            base = name.replace("_gray", "")
            merged.setdefault(base, []).append(score)

        for name, score in gray_scores.items():
            base = name.replace("_gray", "")
            merged.setdefault(base, []).append(score)

        # --- 6. Average final ---
        final_results = []
        for base, values in merged.items():
            avg_score = sum(values) / len(values)
            final_results.append((base, avg_score))

        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)

        return final_results[:top_k]