from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
print('\n'.join(f'"{req}",' for req in requirements))
    
setup(
    name="modlee_pypi",
    version="0.1",
    description="beta",
    packages=find_packages(),
    # packages=["modlee_pypi"],
    python_requires=">=3.8",
    install_requires=[],
    # install_requires=requirements,
    # install_requires=[
    #     "cython==3.0.0",
    #     "keras==2.13.1",
    #     "lightning==2.0.6",
    #     "mlflow==2.5.0",
    #     "numpy==1.24.3",
    #     "pandas==2.0.3",
    #     "pytorch_lightning==2.0.6",
    #     "scikit_learn==1.3.0",
    #     "scipy==1.11.1",
    #     "setuptools==67.7.2",
    #     # "tensorflow",
    #     # "torch==2.0.1",
    #     "torchmetrics==0.11.4",
    #     "torchvision",
    #     "transformers==4.31.0",][::-1],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/modlee-ai/modlee_pypi/"
)
