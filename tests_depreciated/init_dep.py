# import os
# import importlib
# from . import *

# test = "test"
# # Get the current directory
# current_dir = os.path.dirname(__file__)

# # Get a list of all Python files in the directory
# module_files = [
#     file[:-3]
#     for file in os.listdir(current_dir)
#     if file.endswith(".py") and file != "__init__.py"
# ]

# # Import each module dynamically
# for module_file in module_files:
#     module = importlib.import_module(f"{__package__}.{module_file}")
