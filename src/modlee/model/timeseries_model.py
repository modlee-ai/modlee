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

class TimeseriesClassificationModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with timeseries-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TimeseriesClassificationModleeModel constructor.

        """
        modality = 'timeseries'
        task = "classification"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class TimeseriesRegressionModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with timeseries-regression-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TimeseriesRegressionModleeModel constructor.

        """
        modality = 'timeseries'
        task = "regression"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )