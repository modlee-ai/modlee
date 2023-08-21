#
import unittest

import modlee
from modlee.api_client import ModleeAPIClient
# 
import inspect

'''
To properly run these tests, the modlee_api server must be running
in modlee_api/modlee_api, run:
python3 app.py
'''

endpoint = "http://127.0.0.1:5000/"
dummy_endpoint = "http://9.9.9.9:9999"

class ModleeAPIClientTest(unittest.TestCase):
    client = ModleeAPIClient(endpoint="http://127.0.0.1:5000/")
    dummy_client = ModleeAPIClient(endpoint=dummy_endpoint)
    
    def setUpModule(self):
        # client = 
        pass
    def tearDownModule(self):
        pass
    
    def test_get(self):
        response = self.client.get()
        print(response)
        assert 200<=response.status_code<300, "Ensure that server is running"
        # assert server.ava
    
    def test_available(self):
        '''
        Server should be available
        '''
        assert self.client.available, "Server not available, ensure that it is running"
        
    def test_dummy_available(self):
        '''
        Dummy should not be available
        '''
        assert self.dummy_client.available==False, "Disconnected client server should not be available"
    
    def test_get_modlee_attrs(self):
        '''
        Test getting functions
        '''
        attrs_to_get = [
            'hello_world',
            'get_code_text',
            'rep.Rep',
            'data_stats.DataStats',
        ]
        for attr_to_get in attrs_to_get:
            response = self.client.get_attr(attr_to_get)
            assert 200 <= response.status_code < 400, f"Error retrieving {attrs_to_get}: {response.content}"
            
    def test_get_attrs_fail(self):
        '''
        Trying to get random attributes should return none
        '''
        attrs_to_fail = [f"fail_attr_{i}" for i in range(3)]
        for attr_to_fail in attrs_to_fail:
            response = self.client.get_attr(attr_to_fail)
            assert response is None, f"Should not have gotten attribute {attr_to_fail}"
            
    def test_disconnected(self):
        '''
        Fail to get a response from a dummy endpoint
        '''
        dummy_endpoint = "http://9.9.9.9:9999"
        response = self.dummy_client.get()
        assert response is None, \
            f"Should not have gotten a non-error status code from {dummy_endpoint}"
        
    def test_callable(self):
        '''
        Get callable objects (functions or classes)
        '''
        callables_to_get = [
            'hello_world',
            'data_stats.DataStats',
            'get_code_text',
            'get_code_text_for_model'
        ]
        for callable_to_get in callables_to_get:
            response = self.client.get_callable(callable_to_get)
            assert callable(response), f"Could not retrieve callable object {callable_to_get}"
            # function = 
            # response()
        
    def test_fail_to_get_modules(self):
        '''
        Cannot pickle modules so these should fail
        '''
        modules_to_get = [
            'data_stats',
            'utils',
            'rep'
        ]
        for module_to_get in modules_to_get:
            response = self.client.get_attr(module_to_get)
            # assert response.status_code >= 400, f"Should not have gotten module {module_to_get}"
            assert response is None, f"Should not have gotten module {module_to_get}"
        
