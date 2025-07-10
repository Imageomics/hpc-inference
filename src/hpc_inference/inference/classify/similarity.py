import torch
import torch.nn.functional as F

def cosine_similarity(embedding_1, embedding_2, device="cuda"):

    # Check data types
    if not isinstance(embedding_1, torch.Tensor) or not isinstance(embedding_2, torch.Tensor):
        raise TypeError("Both embeddings must be PyTorch tensors.")
    if embedding_1.dtype != torch.float32 or embedding_2.dtype != torch.float32:
        print("Warning: Embeddings should be of type float32 for cosine similarity.")
        embedding_1 = embedding_1.to(torch.float32)
        embedding_2 = embedding_2.to(torch.float32)
        print("Converted embeddings to float32.")
    # Check dimensions
    if embedding_1.dim() != 2 or embedding_2.dim() != 2:
        raise ValueError("Both embeddings must be 2D tensors (batch_size, embedding_dim).")
    if embedding_1.size(1) != embedding_2.size(1):
        raise ValueError("Both embeddings must have the same embedding dimension.")
    # Normalize embeddings
    embedding_1 = embedding_1.to(device)
    embedding_2 = embedding_2.to(device)

    embedding_1 = F.normalize(embedding_1, dim=-1)
    embedding_2 = F.normalize(embedding_2, dim=-1)

    # Dot product
    return embedding_1 @ embedding_2.T

def get_predictions(
        input_embeddings, class_embeddings,
        class_labels, uuid_list = None,
        device="cuda"
    ):
    
    similarity = cosine_similarity(input_embeddings, class_embeddings, device=device)

    pred_scores, pred_indices = similarity.max(dim=1)
    
    if uuid_list is not None:
        predictions = {
            "uuids": uuid_list,
            "pred_label": [class_labels[idx] for idx in pred_indices.cpu()],
            "pred_score": pred_scores.cpu().tolist()
        }
    else:
        predictions = {
            "predicted_labels": [class_labels[idx] for idx in pred_indices.cpu()],
            "scores": pred_scores.cpu().tolist()
        }
    
    del similarity, pred_scores, pred_indices
    return predictions
        


