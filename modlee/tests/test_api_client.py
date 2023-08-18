#
import unittest

import modlee
from modlee.api_client import ModleeAPIClient
# 
import inspect
flask_url = 'http://127.0.0.1:5000/'
# 
client = ModleeAPIClient(endpoint="http://127.0.0.1:5000/")

class ModleeAPIClientTest(unittest.TestCase):
    client = ModleeAPIClient(endpoint="http://127.0.0.1:5000/")
    def setUpModule(self):
        # client = 
        pass
    def tearDownModule(self):
        pass
    
    def test_get(self):
        response = self.client.get()
        assert 200<=response.status_code<300, "Ensure that server is running"
    
    
    def test_get_functions(self):
        '''
        Functions 
        '''
        functions_to_get = [
            'hello_world',
            'get_code_text',
        ]
        for function_to_get in functions_to_get:
            response = self.client.get_function(function_to_get)
            assert response.status_code < 400, f"Error retrieving {function_to_get}"
        
        
    def test_get_modules(self):
        '''
        Cannot pickle modules so these should fail
        '''
        modules_to_get = [
            'data_stats',
            'utils'
        ]
        for module_to_get in modules_to_get:
            response = self.client.get_function(module_to_get)
            assert response.status_code >= 400, f"Should not have gotten module {module_to_get}"
        
