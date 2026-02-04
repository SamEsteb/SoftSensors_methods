import torch
import warnings

# Filtramos la advertencia para que no moleste en la prueba
warnings.filterwarnings("ignore", category=UserWarning)

print("Intentando usar la GPU...")

try:
    # 1. Crear tensores en la GPU
    x = torch.randn(1000, 1000).to('cuda')
    y = torch.randn(1000, 1000).to('cuda')
    
    # 2. Hacer una operación pesada (multiplicación de matrices)
    z = torch.matmul(x, y)
    
    print(f"¡ÉXITO! Operación completada.")
    print(f"Dispositivo usado: {torch.cuda.get_device_name(0)}")
    print(f"Resultado (tamaño): {z.shape}")
    
except Exception as e:
    print(f"FALLO CRÍTICO: {e}")