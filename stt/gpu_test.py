import torch

print("=" * 60)
print("GPU TEST")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("PyTorch CUDA    :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print(
        "VRAM            :",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

    x = torch.randn(
        5000,
        5000,
        device="cuda"
    )

    y = x @ x

    torch.cuda.synchronize()

    print("GPU computation : SUCCESS")
    print(
        "Allocated VRAM  :",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )

else:
    print("ERROR: CUDA is not available")