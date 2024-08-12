import pytest
import modlee
from modlee.utils import discretize
from .conftest import model_from_args

def test_discretize():

    n = 0.234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.00234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 2.34
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 30143215
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = [3.3, 32144321, 0.032]
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = (1, 23)
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = "test"
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.0005985885113477707
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))


@pytest.mark.parametrize("modality_task_kwargs", [
    ("image", "classification", {"num_classes":10}),
    ("image", "segmentation", {"num_classes":10}),
    ])
def test_get_modality_task(modality_task_kwargs):
    modality, task, kwargs = modality_task_kwargs
    model = model_from_args(modality_task_kwargs)
    
    parsed_modality, parsed_task = modlee.utils.get_modality_task(model)
    assert modality == parsed_modality
    assert task == parsed_task
    