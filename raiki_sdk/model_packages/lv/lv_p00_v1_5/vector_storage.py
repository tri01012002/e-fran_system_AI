import pickle
from PIL import Image
import torch
import torch.nn.functional as F

def load_database(load_path, device ):
    """Load database embeddings"""
    with open(load_path, 'rb') as f:
        db_data = pickle.load(f)
    embeddings_db = {
        name: [emb.to(device) for emb in emb_list]
        for name, emb_list in db_data['embeddings'].items()
    }
    return embeddings_db


class VectorStorageManager:
    _instance = None 
    _embeddings_db = None 

    def __new__(cls, model=None, transform=None, vector_db_file = None, device='cuda'):
        if cls._instance is not None:
            return cls._instance

        instance = super(VectorStorageManager, cls).__new__(cls)
        cls._instance = instance  
        return instance

    def __init__(self, model=None, transform=None, vector_db_file=None,  device='cuda'):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model = model.to(device)
        self.model.eval()
        self.transform = transform
        self.device = device

        # Load embeddings chỉ 1 lần duy nhất
        if VectorStorageManager._embeddings_db is None:
            VectorStorageManager._embeddings_db = load_database(vector_db_file,device)
        self.embeddings_db = VectorStorageManager._embeddings_db
        self.idx_to_class = {}

        self._initialized = True  # Đánh dấu đã khởi tạo

    # ---------------- EMBEDDING EXTRACTION ----------------
    def extract_embedding_from_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """Trích xuất embedding từ ảnh PIL"""
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_tensor)
        return emb.squeeze(0)

    # ---------------- COSINE SIMILARITY ----------------
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Tính cosine similarity giữa hai vector"""
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        return torch.dot(emb1, emb2).item()

    
    def recognize_pil(self, pil_image: Image.Image, top_k: int = 3, N_closest: int = 3, return_all_scores: bool = False) -> list[tuple[str, float]]:
        """Nhận dạng khuôn mặt từ ảnh PIL"""
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


    def search_vector(self, query_embedding: torch.Tensor, topk: int = 3, N_closest: int = 3, return_all_scores: bool = False) -> list[tuple[str, float]]:
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

        return sorted_results[:topk]        
