#
import unittest
import yaml

import os
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

# locale endpoing
ENDPOINT = "http://127.0.0.1:7070/"
# remote endpoint
# ENDPOINT = "http://modlee.pythonanywhere.com"
dummy_endpoint = "http://9.9.9.9:9999"

modlee_dev_available = False
try: 
    importlib.import_module('modlee_dev')
    modlee_dev_available = True
except ModuleNotFoundError as e:
    modlee_dev_available = False
    
 
with open('./test_retriever.yaml','r') as test_retriever_file:
    ret_dict = yaml.safe_load(test_retriever_file)
globals().update(dict(
    mlruns_dirs=ret_dict['mlruns_dirs'],
    run_dirs=ret_dict['run_dirs']
))
assert len(run_dirs)>0, f"No run_dirs defined in ./test_retriever.yaml"
    
    
class ModleeAPIClientTest(unittest.TestCase):
    client = ModleeAPIClient(
        endpoint=ENDPOINT,
        api_key='local'
        )
    unauthorized_client = ModleeAPIClient(
        endpoint=ENDPOINT,
        api_key='unauthorized'
    )
    dummy_client = ModleeAPIClient(endpoint=dummy_endpoint)
    
    def setUpModule(self):
        pass
    def tearDownModule(self):
        pass
    
    def test_get(self):
        response = self.client.get()
        assert 200<=response.status_code<300, "Ensure that server is running"
    
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
            'data_stats.DataStats',
            'get_code_text',
            'get_code_text_for_model'
        ]
        for callable_to_get in callables_to_get:
            response = self.client.get_callable(callable_to_get)
            assert callable(response), f"Could not retrieve callable object {callable_to_get}"
        
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
            
    def test_get_callable_from_script(self):
        '''
        Get scripts as raw *.py files
        '''
        scripts_to_get = [
            'data_stats',
            'model_text_converter',
            'exp_loss_logger'
        ]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.client.get_module(script_to_get)
            assert response is not None, f"Did not receive response, get_module({script_to_get}) returned None"
            # if modlee_dev_available:
                # try:
            exec(response,{},locals())
                # except:
                    # assert False, "Could not execute callable even though modlee_dev package available"
                

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

    def test_send_file(self):
        files_paths = [
            ["./test_file.txt", "./test_file.txt"]
        ]

        for file_path in files_paths:
            [file_to_send, path_to_save] = file_path
            response = self.client.post_file(os.path.abspath(file_to_send), path_to_save)
            assert response, f"Could not post {file_path}"
            
    def test_save_run(self):
        
        for run_dir in run_dirs:
            response = self.client.save_run(run_dir)
            assert response, f"Client {self.client.api_key} could not save {run_dir}"

    def test_unauth_save_run(self):
        for run_dir in run_dirs:
            response = self.unauthorized_client.save_run(run_dir)
            assert response is False, f"Unauthorized client should not have saved {run_dir}"
