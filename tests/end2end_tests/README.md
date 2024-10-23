# Modlee Tests

## End-to-end testing

in root directory

### Dummy data

```
python3 -m pytest tests/end2end_tests/test_tabular_classification.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_forecasting.py -k test
python3 -m pytest tests/end2end_tests/test_image_classification.py -k test
python3 -m pytest tests/end2end_tests/test_image_segmentation.py -k test

python3 -m pytest tests/end2end_tests/test_tabular_regression.py -k test
```

### Real data

```
python3 tests/end2end_tests/test_tabular_classification_diabetes.py
python3 tests/end2end_tests/test_image_classification_mnist.py
python3 -m pytest tests/end2end_tests/test_tabular_regression_real_data.py -k test
```