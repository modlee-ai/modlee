import requests
import pickle

DEFAULT_ENDPOINT = "http://127.0.0.1:5000"

class ModleeAPIClient(object):
    def __init__(self,endpoint=DEFAULT_ENDPOINT,*args,**kwargs):
        self.endpoint = endpoint
        
    def get(self,route=""):
        return requests.get(
            f"{self.endpoint}/{route}"
        )
        
    def get_function(self,route=""):
        return self.get(f"modlee/{route}")
        
        