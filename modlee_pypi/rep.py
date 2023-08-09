from importlib.machinery import SourceFileLoader
import json

from mlflow.client import MlflowClient
from mlflow.entities import Run

client = MlflowClient()

class Rep(Run,):
    artifacts = ['modlee_model.py','data_stats']
    
    def __init__(self,run=None,**kwargs):
        self.__dict__.update(**kwargs)
        if 'run_kwargs' in kwargs:
            Run.__init__(self, **kwargs['run_kwargs'])
        
    @classmethod
    def from_run(cls,run,**kwargs):
        run_keys = ['info','data','inputs']
        return cls(run_kwargs={f"run_{k}":getattr(run,k) for k in run_keys})

    def _init_model(self):
        model_path = self._get_artifact('model.py')
        self._modlee_model_module = SourceFileLoader(
            'modlee_mod',model_path,).load_module()
        self._model = self._modlee_model_module.ModleeModel()
        
    def _init_data_stats(self):
        data_stats_path = self._get_artifact('data_stats')
        with open(data_stats_path,'r') as data_stats_file:
            self._data_stats = json.load(data_stats_file)
        
    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._init_model()
        return self._model      
    
    @property
    def data_stats(self):
        if not hasattr(self, '_data_stats'):
            self._init_data_stats()
        return self._data_stats        
    
    def _get_artifact(self, artifact):
        if not getattr(self, 'info', True): 
            raise Exception('Rep not initialized with a run object')
        return client.download_artifacts(self.info.run_id,
            artifact)