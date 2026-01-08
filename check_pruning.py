import sys
try:
    import torch_pruning as tp
    print("Import torch_pruning Success")
except ImportError as e:
    print(f"Import torch_pruning Failed: {e}")

try:
    import thop
    print("Import thop Success")
except ImportError as e:
    print(f"Import thop Failed: {e}")
