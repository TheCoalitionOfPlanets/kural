import torch
from transformers import AutoModel

MODEL_PATH = "./models"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

print("Loading SraVaani...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = model.to(device)
model.eval()
model._ensure_loaded()

print("Model loaded successfully!")
print("Running on:", device)

if torch.cuda.is_available():
    print(
        "VRAM allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )