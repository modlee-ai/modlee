""" 
Recommender for models.
"""
from .recommender import *
from .image_recommender import *


def from_modality_task(modality, task, *args, **kwargs):
    RecommenderObject = globals().get(
        f"{modality.capitalize()}{task.capitalize()}Recommender", None
    )
    assert RecommenderObject is not None, f"No Recommender for {modality} {task}"
    return RecommenderObject(*args, **kwargs)
