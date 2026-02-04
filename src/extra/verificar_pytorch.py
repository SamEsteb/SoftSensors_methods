import torch
print(f"PyTorch versión: {torch.__version__}")
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
else:
    print("¡OJO! Estás usando CPU. Revisa la instalación de PyTorch.")