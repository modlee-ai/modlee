
## Guide: Extending Modlee for Tabular Data Use Cases

This guide provides an in-depth approach to extending Modlee to handle tabular data for classification tasks. It includes steps for dataset discovery, preprocessing, implementing and testing metafeatures, and validating functionality to ensure effective integration of a new tabular data modality into Modlee.

### 1. **Find and Preprocess Tabular Datasets**

**Objective:**

- Identify suitable tabular datasets and preprocess them for use with Modlee.

**Steps:**

1.  **Search for Datasets:**
    
    - **Explore Dataset Platforms:** Start by exploring popular platforms like Kaggle to find relevant tabular datasets. These platforms offer a wide range of datasets with various characteristics, which can be crucial for testing different scenarios.
  
    - **Review Dataset Discussions:** Participate in discussions and forums such as [this Kaggle discussion](https://www.kaggle.com/discussions/general/443626) to get recommendations and insights on datasets that might be suitable for your needs. This can provide you with examples of datasets used by others and new sources to consider.
  
2.  **Download and Load Datasets:**
    
    - **Obtain Data Files:** Download datasets in formats like CSVs. Ensure the dataset’s documentation is reviewed to understand its structure and contents.
    - **Load into DataFrames:** Use `pandas` to load the datasets into `DataFrames`, which are easy to manipulate and analyze. This step allows you to inspect the data for initial cleaning and preprocessing.
    
3.  **Preprocess Datasets:**
    
    - **Handle Missing Values:** Address missing values by removing rows/columns with excessive missing values. This ensures the dataset is complete and usable.
        
    - **Encode Categorical Variables:** Convert categorical variables into numerical formats using techniques like one-hot encoding or label encoding. This step is necessary because most machine learning algorithms require numerical inputs.
        
    - **Normalize/Standardize Data:** Apply scaling techniques to ensure all features contribute equally to the model’s performance. Standardizing or normalizing features can improve convergence during training.

### 2. **Prepare and Use a DataLoader**

**Objective:**

- Prepare a DataLoader to efficiently load tabular data into batches for model training.
  
**Steps:** 

1. **Create a Custom Dataset:**

    - **Define Dataset Class:** Implement a custom `Dataset` class for tabular data by extending `PyTorch’s Dataset` class. This class should handle loading data and providing samples.

    ```python
    class TabularDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    ```

2. **Create DataLoader Instances:**

    - **Initialize DataLoader:** Use `PyTorch’s DataLoader` to create data loaders for training, validation, and testing. This helps in managing batch sizes and shuffling data during training.

    ```python
    dataset = TabularDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    ```

### 3. **Define Tabular Data Metafeatures**

**Objective:**

- Create a `TabularDataMetafeatures` class to compute and manage metafeatures for tabular data.

- **Compute Base Data Metafeatures:**
    - **Initial Implementation:** Initially, define a simple subclass of `DataMetafeatures` to ensure that the base functionality works correctly with tabular data.This serves as a foundation before adding more complex features. 
        
        ```python
        class TabularDataMetafeatures(DataMetafeatures):
            pass
        ```
        
    - **Verify Basic Functionality:** Test this implementation with a sample dataloader to ensure that the base class is correctly handling tabular data. Confirm that no errors occur and that the class is integrating with Modlee as expected.

### 4. **Test the Implementation**

**Objective:**

- Validate the functionality of the base metafeatures and that it works correctly with various tabular datasets.

**Steps:**

1.  **Check Base Functionality:**
    
    - **Ensure Base Functionality:** Before adding statistical features, verify that the base `DataMetafeatures` class operates correctly with the dataloader. Ensure that the core functionality is solid and that the class correctly processes tabular data.
  
2.  **Create Test Cases:**
    
    - **Write Tests for Base Functionality:** Use `pytest` to create test cases that validate the base metafeatures. These tests ensure that the initial implementation correctly integrates with Modlee and handles data as expected.
    - Reference existing `pytest` implementations for base data metafeature calculations.
    - Choose which base metafeatures to test out for your modality. For the tabular use case, `mfe`, `properties`, and `features` were uesd.

3.  **Run Tests:** Run the `pytest` to ensure that the base data meta features were calculated and that we are passing the test cases.  

### 5. **Implement Statistical Metafeatures**

**Objective:**

- Extend the `TabularDataMetafeatures` class to include statistical summaries such as mean, variance, and quantiles.

   - **Extend Functionality:** Add methods to compute statistical measures for each batch element. This includes calculations such as mean, variance, and quantiles, which provide insights into the data distribution and variability.
        
        ```python
        import pandas as pd
        import torch
        
        class TabularDataMetafeatures(DataMetafeatures):
            def __init__(self, dataloader, *args, **kwargs):
                super().__init__(dataloader, *args, **kwargs)
                self.stats_rep = self.get_features()
                self.features.update(self.stats_rep)
            
            def get_features(self):
                stats_rep = {}
                for idx, element in enumerate(self.batch_elements):
                    if isinstance(element, torch.Tensor):
                        np_element = element.numpy()
                        stats = self.calculate_statistical_summary(np_element)
                        for key, value in stats.items():
                            stats_rep[f'batch_element_{idx}_{key}'] = value
                return stats_rep
        
            def calculate_statistical_summary(self, data):
                df = pd.DataFrame(data)
                summary = {
                    'mean': df.mean().tolist(),
                    'median': df.median().tolist(),
                    'variance': df.var().tolist(),
                    'std_dev': df.std().tolist(),
                    'min': df.min().tolist(),
                    'max': df.max().tolist(),
                    'range': (df.max() - df.min()).tolist(),
                    'quantiles_25': df.quantile(0.25).tolist(),
                    'quantiles_50': df.quantile(0.50).tolist(),
                    'quantiles_75': df.quantile(0.75).tolist()
                }
                return summary
        ```
           
        

### 6. **Test the Statistical Meta Features Implementation**

**Objective:**

- Validate the functionality of the statistical metafeatures through comprehensive testing.

  - **Test Statistical Metafeatures:** Extend the test cases to validate the statistical measures implemented in `TabularDataMetafeatures`. Ensure that statistical summaries are correctly computed and included in the metafeature set.
        
        ```python       
        import pytest
        
        @pytest.mark.experimental
        class TestTabularDataMetafeatures:
         @pytest.mark.parametrize('get_dataloader_fn', TABULAR_LOADERS.values())
            def test_tabular_dataloader(self, get_dataloader_fn):
                tabular_mf = TabularDataMetafeatures(get_dataloader_fn())
                self._check_has_metafeatures_tab(tabular_mf)
                self._check_statistical_metafeatures(tabular_mf)
        
            def _check_has_metafeatures_tab(self, mf):
                metafeature_types = ['mfe', 'properties', 'features']
                for metafeature_type in metafeature_types:
                    assert hasattr(mf, metafeature_type)
                    assert isinstance(getattr(mf, metafeature_type), dict)
                    assert not any(isinstance(v, dict) for v in getattr(mf, metafeature_type).values())
        
            def _check_statistical_metafeatures(self, mf):
                statistical_metafeatures = ['mean', 'std_dev', 'median', 'min', 'max', 'range']
                features = getattr(mf, 'features')
                for feature in statistical_metafeatures:
                    assert any(feature in key for key in features)
                assert not any(isinstance(v, dict) for v in features.values())
        ```

- **Run Test Cases:**
    
    - **Execute Tests:** Run the test cases to ensure that both base and statistical metafeatures are functioning correctly. Address any issues that arise during testing to ensure that the implementation is robust and accurate.

By following this detailed guide, you will be able to extend Modlee to support tabular data use cases effectively. This ensures that you can document, analyze, and optimize your tabular datasets for improved machine learning experiments.