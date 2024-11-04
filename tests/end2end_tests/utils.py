import os
import json
import math
import importlib
import torch

def check_artifacts(artifacts_path):
    """
    Checks for specific files in the artifacts directory, opens them, 
    and verifies the variable types of their contents.

    Args:
        artifacts_path (str): The path to the artifacts directory.

    Returns:
        None
    """

    print('--- checking experiment artifacts')

    specific_elements = ['model_metafeatures', 'model_size', 'model_summary.txt', 'checkpoints', 'model.py', 'cached_vars', 'stats_rep', 'model', 'model_graph.py', 'model_graph.txt', 'data_metafeatures']

    # List the files in the artifacts directory
    artifacts = os.listdir(artifacts_path)

    # Assert that specific elements are in artifacts
    for element in specific_elements:
        assert element in artifacts, f"Missing expected artifact: {element}"

    for artifact in artifacts:
        if artifact in specific_elements:
            # Construct the full path to the file
            file_path = os.path.join(artifacts_path, artifact)

            try:
                # Open the file and check its contents
                with open(file_path, 'r') as file:
                    content = file.read()
            except:
                # print('not a file to be read: ',artifact)
                continue
        
            if artifact in ['model_metafeatures', 'data_metafeatures','cached_vars']:
                # Assuming the content is in JSON format
                variable = json.loads(content)

                # Count the number of NaN, str, float, and int values
                num_nans = sum(1 for value in variable.values() if isinstance(value, float) and math.isnan(value))
                num_str = sum(1 for value in variable.values() if isinstance(value, str))
                num_float = sum(1 for value in variable.values() if isinstance(value, float))
                num_int = sum(1 for value in variable.values() if isinstance(value, int))
                num_list = sum(1 for value in variable.values() if isinstance(value, list))
                num_tuple = sum(1 for value in variable.values() if isinstance(value, tuple))

                # Assert that the total count of str, float, and int is not zero
                assert (num_str + num_float + num_int + num_tuple + num_list) != 0, "The total count of str, float, and int is zero."

                # Print counts for verification
                print(f"Counts in {artifact}: NaNs: {num_nans}, Strings: {num_str}, Floats: {num_float}, Integers: {num_int}, Tuple: {num_tuple}, List: {num_list}")
            elif artifact == 'model_size':
                assert type(int(content))==int
            elif artifact == 'model_graph.py':
                # Load the module from the file
                spec = importlib.util.spec_from_file_location("model_graph", os.path.join(artifacts_path,artifact))
                model_graph = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_graph)

                # Check if the Model class is in the module
                assert hasattr(model_graph, 'Model')
    print('passed experiment artifact check')


def get_device():
    """
    Dynamically selects the best available device for training:
    - Intel iGPU (xpu) if available with PyTorch oneAPI support (requires Intel® Extension for PyTorch*)
    - Apple Silicon (mps) if available on Apple M1/M2 chips
    - CUDA (cuda) if available on NVIDIA GPUs
    - Defaults to CPU if none are available.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using device: Apple Silicon (mps)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: NVIDIA GPU (cuda)")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using device: Intel iGPU (xpu)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device
