""" 
Configuration variables.
"""
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.absolute()
MODLEE_DIR = ROOT_DIR / "modlee"
TMP_DIR = ROOT_DIR / "tmp"
MLRUNS_DIR = ROOT_DIR / "mlruns"

LOCAL_ORIGIN = "http://127.0.0.1:7070"
SERVER_HOSTNAME = "http://ec2-3-84-155-233.compute-1.amazonaws.com"
SERVER_ORIGIN = f"{SERVER_HOSTNAME}:7070"
RECOMMENDER_ORIGIN = f"{SERVER_HOSTNAME}:6060"
