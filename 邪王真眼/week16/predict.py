import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import torch.nn.functional as F

from models.models import get_model
from datasets.datasets import get_dataset
from configs.config import load_train_config


def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)


def generate_text(model, device, id2char, vocab_size, gen_length=20, k=50):
    seq = [random.randint(2, vocab_size - 1)]
    print(f"起始字: {id2char[seq[0]]}", end="", flush=True)
    with torch.no_grad():
        for _ in range(gen_length - 1):
            input_ids = torch.tensor([seq], dtype=torch.long).to(device)
            mask = torch.ones_like(input_ids)
            logits = model(input_ids, mask)["out"]
            last_logits = logits[:, :, -1]
            filtered_logits = top_k_logits(last_logits, k)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            seq.append(next_token)
            print(id2char.get(next_token, ""), end="", flush=True)
    print()


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week16.py")
    cfg['batch_size'] = 1
    _, val_loader = get_dataset(cfg)
    id2char = {idx: char for char, idx in val_loader.dataset.vocab.items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)
    model_path = "./邪王真眼/week16/result/nano_gpt.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for _ in range(10):
        generate_text(model, device, id2char, cfg['vocab_size'], gen_length=20, k=50)
