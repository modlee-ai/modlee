# Examples
These notebooks are not typical Jupyter`.ipynb` notebooks, but instead are plain Python`.py` files with the [`# %%` cell separator syntax recognized in VS Code](https://code.visualstudio.com/docs/python/jupyter-support-py).
Keeping them as plain Python enables execution as either scripts (`python3 script.py`) or cell-by-cell using `Shift+Enter` in the editor.

## `simple_torch.py`
[`simple_torch.py`](./simple_torch.py) is a simple working example.
The script defines the model, trains, and saves the logs to `./mlruns` relative to the examples directory.