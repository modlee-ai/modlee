from torchtext import models as ttm

from modlee.utils import text_loaders

TEXT_MODELS = [
    # ttm.FLAN_T5_BASE,
    # ttm.FLAN_T5_BASE_ENCODER,
    # ttm.FLAN_T5_BASE_GENERATION,
    ttm.ROBERTA_BASE_ENCODER,
    ttm.ROBERTA_DISTILLED_ENCODER,
    # ttm.T5_BASE,
    # ttm.T5_BASE_ENCODER,
    # ttm.T5_SMALL,
    # ttm.T5_SMALL_ENCODER,
    # ttm.T5_SMALL_GENERATION,
    ttm.XLMR_BASE_ENCODER,
    # ttm.XLMR_LARGE_ENCODER, # Too large for M2 MacBook Air?
]

TEXT_MODALITY_TASK_MODEL = []
for model in TEXT_MODELS:
    TEXT_MODALITY_TASK_MODEL.append(("text", "classification", model))
