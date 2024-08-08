from modlee.model import ModleeModel, DataMetafeaturesCallback, ModelMetafeaturesCallback
from modlee import data_metafeatures as dmf, model_metafeatures as mmf

class TextModleeModel(ModleeModel):
    def __init__(self, task="classification", num_classes=None, *args, **kwargs):
        if not num_classes:
            raise AttributeError("Must provide argument for num_classes")
        else:
            self.num_classes = num_classes
        self.task = task
        vars_cache = {"num_classes": num_classes, "task": task}
        ModleeModel.__init__(self, kwargs_cache=vars_cache, *args, **kwargs)

    def configure_callbacks(self):
        base_callbacks = super().configure_callbacks()
        text_data_mf_callback = DataMetafeaturesCallback(
            DataMetafeatures=dmf.TextClassificationDataMetafeatures
        )
        return [*base_callbacks, text_data_mf_callback]

class TextClassificationModleeModel(TextModleeModel):
    def configure_callbacks(self):
        text_model_mf_callback = ModelMetafeaturesCallback(
            ModelMetafeatures=mmf.TextClassificationModelMetafeatures
        )
        return [*super().configure_callbacks(), text_model_mf_callback]