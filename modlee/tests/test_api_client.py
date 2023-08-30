#
import unittest

import modlee
from modlee.api_client import ModleeAPIClient
import importlib
# 
import inspect

'''
To properly run these tests, the modlee_api server must be running
in modlee_api/modlee_api, run:
python3 app.py
'''

ENDPOINT = "http://127.0.0.1:5000/"
dummy_endpoint = "http://9.9.9.9:9999"

modlee_dev_available = False
try: 
    importlib.import_module('modlee_dev')
    modlee_dev_available = True
except ModuleNotFoundError as e:
    modlee_dev_available = False
    
    

class ModleeAPIClientTest(unittest.TestCase):
    client = ModleeAPIClient(
        endpoint=ENDPOINT,
        api_key='user1'
        )
    unauthorized_client = ModleeAPIClient(
        endpoint=ENDPOINT,
        api_key='unauthorized'
    )
    dummy_client = ModleeAPIClient(endpoint=dummy_endpoint)
    # client.login('user1')
    # client.login('failuser')
    
    def setUpModule(self):
        # client = 
        pass
    def tearDownModule(self):
        pass
    
    def test_get(self):
        response = self.client.get()
        assert 200<=response.status_code<300, "Ensure that server is running"
        # assert server.ava
    
    def test_available(self):
        '''
        Server should be available
        '''
        assert self.client.available, "Server not available, ensure that it is running"
        
    @unittest.skip
    def test_dummy_available(self):
        '''
        Dummy should not be available
        '''
        assert self.dummy_client.available==False, "Disconnected client server should not be available"
    
    @unittest.skipIf(modlee_dev_available==False, "modlee_dev not installed to test pypi-only env, skipping")
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

    # @unittest.skipIf(modlee_dev_available==False)            
    def test_get_attrs_fail(self):
        '''
        Trying to get random attributes should return none
        '''
        attrs_to_fail = [f"fail_attr_{i}" for i in range(3)]
        for attr_to_fail in attrs_to_fail:
            response = self.client.get_attr(attr_to_fail)
            assert response is None, f"Should not have gotten attribute {attr_to_fail}"
        
    @unittest.skip
    def test_disconnected(self):
        '''
        Fail to get a response from a dummy endpoint
        '''
        dummy_endpoint = "http://9.9.9.9:9999"
        response = self.dummy_client.get()
        assert response is None, \
            f"Should not have gotten a non-error status code from {dummy_endpoint}"
        
    @unittest.skipIf(modlee_dev_available==False, "modlee_dev not installed to test pypi-only env, skipping")
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
        
    @unittest.skipIf(modlee_dev_available==False, "modlee_dev not installed to test pypi-only env, skipping")
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
        

    def test_get_script(self):
        '''
        Get scripts as raw *.py files
        '''
        scripts_to_get = [
            'data_stats'
        ]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.client.get_script(script_to_get)
            assert response, f"No response received (likely None)"
            assert 200 <= response.status_code < 400, f"Could not get script {script_to_get}"
            # exec(response.content, script_dict)
            # print(script_dict)
            
    def test_get_callable_from_script(self):
        '''
        Get scripts as raw *.py files
        '''
        scripts_to_get = [
            'data_stats',
            'model_text_converter'
        ]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.client.get_module(script_to_get)
            assert response, f"Did not receive response, likely None"
            exec(response,{},locals())

    def test_fail_dummy_gets_callable_from_script(self):
        '''
        Get scripts as raw *.py files
        '''
        scripts_to_get = [
            'data_stats',
            'model_text_converter'
        ]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.unauthorized_client.get_module(script_to_get)
            assert response is None, f"Unauthorized client should not have gotten {script_to_get}"

    