# import os, importlib

# MODALITIES = [
#     os.path.splitext(os.path.basename(f))[0]
#     for f in os.listdir(os.path.dirname(__file__))
#     if "py" in os.path.splitext(f)[-1] and not "__init__" in f
# ]

# for modality in MODALITIES:
#     exec(f"from .{modality} import *")
#     # importlib.import_module(modality)
