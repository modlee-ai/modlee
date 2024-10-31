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
        modality = "image"
        task = "classification"
        vars_cache = {"modality":modality,"task": task}
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
        modality = 'image'
        task = "segmentation"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class ImageRegressionModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with tabular-regression-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        ImageRegressionModleeModel constructor.

        """
        modality = 'image'
        task = "regression"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class ImageImageToImageModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with image-to-image-specific convenience wrappers.
    - Designed for tasks where both input and output are images, such as image translation, super-resolution, or denoising.
    - Calculates data-specific statistics and metrics as needed for image-to-image tasks.
    """

    def __init__(self, *args, **kwargs):
        """
        ImageToImageModleeModel constructor.
        """
        modality = "image"
        task = "image-to-image"
        vars_cache = {"modality": modality, "task": task}
        super().__init__(kwargs_cache=vars_cache, *args, **kwargs)