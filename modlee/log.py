import mlflow

def log(**kwargs):
    mlflow.pytorch.autolog()
    for key,value in kwargs.items():
        if isinstance(value,dict):
            mlflow.log_dict(value,key)
        else:
            mlflow.log_param(key,value)
    pass