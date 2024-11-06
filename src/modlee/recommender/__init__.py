""" 
Recommender for models.
"""
from .recommender import *
from .image_recommender import *
from .tabular_recommender import *
from modlee.utils import class_from_modality_task
from functools import partial

from_modality_task = partial(class_from_modality_task, _class="Recommender")
