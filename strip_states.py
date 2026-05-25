import torch, os

ckpt = torch.load("ckpt_006000.pt", map_location="cpu")

clean = {
    "model": ckpt["model"],
    "config": ckpt["config"],
    "step": ckpt["step"],
    "tokens_seen": ckpt["tokens_seen"],
}

torch.save(clean, "ckpt_006000_weights_only.pt")
print(f"Original: {os.path.getsize('ckpt_006000.pt')/1e9:.2f}GB")
print(f"Stripped: {os.path.getsize('ckpt_006000_weights_only.pt')/1e9:.2f}GB")