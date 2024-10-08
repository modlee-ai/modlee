from .model import *
from .image_model import *
from .tabular_model import *
from .timeseries_model import *
from .recommended_model import *
from . import callbacks
from . import trainer
from modlee.utils import class_from_modality_task
from functools import partial

from_modality_task = partial(class_from_modality_task, _class="Model")
