""" 
Modlee model for images. 
"""

from modlee.model import (
    ModleeModel,
)

TASK_METRIC = {"classification": "Accuracy", "regression": "MeanSquaredError"}

class ImageClassificationModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with image-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        ImageClassificationModleeModel constructor.

        """
        task = "classification"
        vars_cache = {"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class ImageSegmentationModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with image-segmentation-specific convenience wrappers
    - Calculates data-specific data statistics
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        ImageSegmentationModleeModel constructor.

        """
        task = "segmentation"
        vars_cache = {"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )
