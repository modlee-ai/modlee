""" 
Recommender for models.
"""
from .recommender import *
from .image_recommender import *


def from_modality_task(modality, task, *args, **kwargs):
    """
    Return a Recommender object based on the modality and task.
    Currently supports:
    
    - image
    --- classification
    --- segmentation
    - text
    --- classification

    :param modality: The modality as a string, e.g. "image", "text".
    :param task: The task as a string, e.g. "classification", "segmentation".
    :return: The RecommenderObject, if it exists.
    """
    RecommenderObject = globals().get(
        f"{modality.capitalize()}{task.capitalize()}Recommender", None
    )
    assert RecommenderObject is not None, f"No Recommender for {modality} {task}"
    return RecommenderObject(*args, **kwargs)
