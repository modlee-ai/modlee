""" 
Configuration variables.
"""
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.absolute()
MODLEE_DIR = ROOT_DIR / "modlee"
TMP_DIR = ROOT_DIR / "tmp"
MLRUNS_DIR = ROOT_DIR / "mlruns"

LOCAL_ENDPOINT = "http://127.0.0.1:7070"
SERVER_ENDPOINT = "http://ec2-3-84-155-233.compute-1.amazonaws.com:7070"
RECOMMENDER_ENDPOINT = "http://ec2-3-84-155-233.compute-1.amazonaws.com:6060"
