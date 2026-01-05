import sys
import os
print("Starting check...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("Path added.")
try:
    from quantization.utils import get_dataloader
    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
