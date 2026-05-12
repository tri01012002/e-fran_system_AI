import pickle
import torch
from PIL import Image

# ==================== Load Database ====================
def _load_database(load_path, device ):
    """Load database embeddings"""
    with open(load_path, 'rb') as f:
        db_data = pickle.load(f)
    embeddings_db = {
        name: [emb.to(device) for emb in emb_list]
        for name, emb_list in db_data['embeddings'].items()
    }
    names = [name for name, emb_list in db_data['embeddings'].items()]
    names = [name.replace("_rotate","").replace("_gray","").split("-")[0] for name in names]
    set_label  = set()
    for name in names:
        if "_" in name:
            set_label.add(name.split("_")[0])
            set_label.add(name.split("_")[1])
        else:
            set_label.add(name.split("_")[0])
    print(f"Loaded embeddings {len(list(set_label))} classes from:", load_path)

    return embeddings_db

# ==================== Load Model ====================
def _load_weights(num_classes, embedding_size, pretrained=False, model_name='vit_base_patch16_dinov3.lvd1689m', img_size=256, checkpoint_path=None, device='cuda'):
    from .architecture import MViTArcFaceModel

    model = MViTArcFaceModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        pretrained=pretrained, 
        model_name=model_name,
        img_size=img_size
    )
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)
    model = model.eval()
    return model