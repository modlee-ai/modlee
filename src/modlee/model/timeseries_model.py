import torch
from modlee.model import ModleeModel


class TimeseriesForecastingModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with tabular-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TabularClassificationModleeModel constructor.

        """
        modality = 'timeseries'
        task = "forecasting"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )
