import os
import pathlib

def safe_mkdir(target_dir):
    root,ext = os.path.splitext(target_dir)
    # is a file
    if len(ext)>0:
        target_dir = os.path.split(root)
    else:
        target_dir = f"{target_dir}/"
    # if os.path.isfile(target_dir):
    #     target_dir,_ = os.path.split(target_dir.split('.')[0])
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)