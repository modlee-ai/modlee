import pathlib
# pathlib.Path(__file__)
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
MODLEE_DIR = ROOT_DIR / 'modlee'
TMP_DIR = ROOT_DIR / 'tmp'
MLRUNS_DIR = ROOT_DIR / 'mlruns'
