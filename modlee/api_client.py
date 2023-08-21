import requests
# import pickle
# import dill as pickle
import cloudpickle as pickle
import functools

DEFAULT_ENDPOINT = "http://127.0.0.1:5000"

class ModleeAPIClient(object):
    def __init__(self,endpoint=DEFAULT_ENDPOINT,*args,**kwargs):
        self.endpoint = endpoint
        self.get_object = self.get_callable
        self.get_function = self.get_callable
        self.timeout = 3
        
    @property
    def available(self):
        ret = self.get()
        if ret is not None:
            return 200<=ret.status_code<400
        else:
            return False
        
    def get(self,route=""):
        req_url = f"{self.endpoint}/{route}"
        try:
            ret = requests.get(
                req_url,
                timeout=self.timeout
            )
            if ret.status_code > 400:
                ret = None
        except requests.Timeout as e:
            ret = None
        return ret
        
    def get_attr(self,route=""):
        return self.get(f"modlee/{route}")
    
    def get_callable(self,route=""):
        ret = self.get_attr(route)
        if ret is not None:
            ret = pickle.loads(ret.content)
        return ret
        
    def get_script(self,route=""):
        return self.get(f"modleescript/{route}")
    
    def get_module(self,route=""):
        ret = self.get_script(route)
        if ret is not None:
            ret = ret.content
            # _locals = {}
            # exec(ret, {}, _locals)
            # ret = _locals
        return ret
        