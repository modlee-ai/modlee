# Introduction

### Welcome to Modlee!

Thank you for your interest in contributing to Modlee! Please take a moment to review this document before submitting a pull request.

We continue improving Modlee with feedback from the community.
As we get Modlee off the ground, we welcome contributions of any kind: bug reports, feature requests, tutorials, etc.

# Ground Rules

**Please treat others with kindness and respect. No harassment, hate speech, or trolling will be tolerated.**

# Resources

Reference for using the `modlee` package is available at the [Modlee API documentation](https://www.documentation.modlee.ai).

# Getting started

### Submitting a contribution

1. Fork this repository.
2. Commit your changes in your fork.
3. Submit a [pull request](https://github.com/modlee-ai/modlee/pulls).

As a rule of thumb, please bundle small changes (e.g. typographic or grammatic corrections, comments) with larger contributions (e.g. function refactors).

# How to report a bug
### Security vulnerabilities
If you find a security vulnerability, **please do NOT open an issue**. Please report the vulnerability to [brad@modlee.ai](brad@modlee.ai).

### General issues
You can even include a template so people can just copy-paste (again, less work for you).

When filing a bug report as an issue, please include the following information:
1. Versions of your technical stack:
    - Operating system
    - Python
    - Framework (PyTorch)
2. Actions you took — what steps led to the error?
3. Expected functionality — what did you *expect* to happen?
4. Actual functionality - what *actually* happened?

# Feature requests
Please post feature requests as an issue with the following information:

1. Description of the proposed feature.
2. Motivation for the feature (e.g. bug fix, quality-of-life improvement, expanding usability).
3. Rough outline of proposed implementation (e.g. use `x` framework in `y` module).

# Code review process

The core Modlee team will review pull requests by submission order, ideally within a week from submission.

# Community
For more informal and ephemeral discussion with the core team and other users, [please join our Discord server](https://discord.com/invite/m8YDbWDvrF).
We plan to office hours in the future.

# Common Contributions

Here are common ways that you can contribute to Modlee.


## Extending to new modalities, tasks, and applications

We can define a machine learning as a modality (e.g. image, text) applied to a task (e.g. classification, segmentation), e.g. image classification.
The following table shows currently supported and potential applications.

| Task \ Modality               | Image   | Text    | Audio   | Video   | Time Series | Graphs   | Multi-modal |
|------------------------------|---------|---------|---------|---------|-------------|----------|-------------|
| **Classification**           |    :heavy_check_mark:    |   :heavy_check_mark:     |         |         |             |          |             |
| **Regression**               |         |         |         |         |             |          |             |
| **Clustering**               |         |         |         |         |             |          |             |
| **Dimensionality Reduction** |         |         |         |         |             |          |             |
| **Generation**               |         |         |         |         |             |          |             |
| **Object Detection**         |         |         |         |         |             |          |             |
| **Anomaly Detection**        |         |         |         |         |             |          |             |
| **Semantic Segmentation**    |    :heavy_check_mark:    |         |         |         |             |          |             |
| **Instance Segmentation**    |         |         |         |         |             |          |             |
| **Recommendation**           |         |         |         |         |             |          |             |
| **Forecasting**              |         |         |         |         |             |          |             |


The following guide covers how to extend Modlee to a new modality-task application.
Extending the package contributes functionality for others to document experiments, share metafeatures with the Modlee server, and improve the model recommendation technology.

### Help

Join us on [Discord](https://discord.gg/m8YDbWDvrF), and collaborate with us in the 'github-contributors' channel for hands on help in supporting your ML Framework.

### 1) Choose an unsupported modality-task application

Use the above table as a guide when choosing an unsupported application or define a new application.

### 2) Fork modlee > main

Follow the above 'Submitting a contribution' instructions to set up your fork in the way we recommend.

### 3) Document dataset metafeatures
- Review:
    - [ModleeModel class](https://github.com/modlee-ai/modlee/blob/0dc107f74146ecef5fa054e4258f8389d6c952ec/src/modlee/model/model.py#L38): core object for defining a PyTorch model that automatically documents. Our auto-logging capabilities are controlled through [callbacks](https://github.com/modlee-ai/modlee/blob/0dc107f74146ecef5fa054e4258f8389d6c952ec/src/modlee/model/model.py#L95) within the ModleeModel class.
    - [DataMetafeatures class](https://github.com/modlee-ai/modlee/blob/0dc107f74146ecef5fa054e4258f8389d6c952ec/src/modlee/data_metafeatures.py#L472C7-L472C23): core object for reducing datasets into reduced dimension. This produces a comparable representation of your dataset that is essential to preserve as you experiment.
- Create:
    - Try DataMetafeatures on new application: see if the core class works on an example pytorch dataloader for your application. Study how both the dataloader or DataMetafeatures would need to be changed in order to accomodate your new ML Task.
    - Develop subclass of DataMetafeatures: You may need to create a new subclass of DataMetafeatures to support your application. Look to [ImageDataMetafeatures](https://github.com/modlee-ai/modlee/blob/0dc107f74146ecef5fa054e4258f8389d6c952ec/src/modlee/data_metafeatures.py#L646C7-L646C28), as an example for how to do this.
- Test:
    - [Pytest Callbacks](https://github.com/modlee-ai/modlee/blob/0dc107f74146ecef5fa054e4258f8389d6c952ec/tests/test_callbacks.py#L12): add to pytest in tests folder for testing out new modality and task using DatasetMetafeatures, or new subclass. Create random dummy data with the shape of your batch elements from your dataloader, then calculate dataset metafeatures and assert the outputs are desired.

### 4) Create example for application
- Review:
    - [Github Examples](https://github.com/modlee-ai/modlee/tree/main/examples): Review the current examples to get context on others contributions.
- Create:
    - Create an example in the form of an `.ipynb` using our [Automate experiment documentation](https://github.com/modlee-ai/modlee/blob/main/examples/document.ipynb) example as a `template` for yours.
    - Note yourself as the author of the example at the top.
    - Do your best to provide some educational context about your example start, but keep it as simple and clean as possible.    
- Test:
    - Run your example and test that all essential elements about your experiment are documented correctly. Please compare your documented experiment log to that of our [Automate experiment documentation](https://github.com/modlee-ai/modlee/blob/main/examples/document.ipynb) example.
    - TODO Modlee internal: Build testing function to verify that ML examples are documented correctly.

### 5) Modlee review
- Submit a pull request, following our suggestions in the above `Submitting a contribution` section
- Modlee will review internally, test out your example, and verify that you are documenting the application correctly.
- Modlee will merge your pull requeset into `main`, and update our pip package at a convinent time for the community.
- We announce announce support and recognize your contribution publically and within this contributor guide.

### 6) Modlee expands model recommender to application
- Modlee will test and validate our model recommendations based on your newly contributed application.
- Modlee will update `main` to allow for model recommendations for your newly contributed application.
- Modlee will update our pip package at a convinent time for the community.
