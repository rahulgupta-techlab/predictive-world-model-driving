import platform
import sys

print("OS:", platform.platform())
print("Python:", sys.version)
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        test = torch.randn((1024, 1024), device="cuda")
        print("CUDA smoke test:", (test @ test).shape)
except ImportError:
    print("PyTorch not installed.")
try:
    import carla
    print("CARLA API import: OK")
except ImportError:
    print("CARLA API not installed.")
