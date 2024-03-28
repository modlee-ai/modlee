"""
Test modlee.client
"""
#
import unittest, pytest
import yaml

import os
import modlee
from modlee.client import ModleeClient
from modlee.config import LOCAL_ORIGIN, SERVER_ORIGIN
import importlib

#
import inspect

"""
To properly run these tests, the modlee_server must be running
in modlee_server/modlee_api, run:
python3 app.py
"""

# local endpoint
# ENDPOINT = LOCAL_ORIGIN
ENDPOINT = SERVER_ORIGIN
# remote endpoint
# ENDPOINT = "http://modlee.pythonanywhere.com"
dummy_ORIGIN = "http://9.9.9.9:9999"

modlee_dev_available = False
try:
    importlib.import_module("modlee_dev")
    modlee_dev_available = True
except ModuleNotFoundError as e:
    modlee_dev_available = False


run_paths = [os.path.join(os.path.dirname(__file__), "test_mlruns")]

assert os.environ.get("MODLEE_API_KEY"), "MODLEE_API_KEY environment variable not set, cannot test test_client.py"

# class ModleeClientTest(unittest.TestCase):
class TestModleeClient:
    client = ModleeClient(
        endpoint=ENDPOINT,
        api_key=os.environ.get("MODLEE_API_KEY")
        # api_key=''
    )
    unauthorized_client = ModleeClient(endpoint=ENDPOINT, api_key="unauthorized")
    dummy_client = ModleeClient(endpoint=dummy_ORIGIN)

    def setUpModule(self):
        pass

    def tearDownModule(self):
        pass

    def test_get(self):
        response = self.client.get()
        assert 200 <= response.status_code < 300, "Ensure that server is running"

    def test_available(self):
        """
        Server should be available
        """
        assert self.client.available, "Server not available, ensure that it is running"

    @unittest.skip
    def test_dummy_available(self):
        """
        Dummy should not be available
        """
        assert (
            self.dummy_client.available == False
        ), "Disconnected client server should not be available"

    @unittest.skipIf(
        modlee_dev_available == False,
        "modlee_dev not installed to test pypi-only env, skipping",
    )
    def test_get_modlee_attrs(self):
        """
        Test getting functions
        """
        attrs_to_get = ["get_code_text", "rep.Rep", "data_metafeatures.DataMetafeatures"]
        for attr_to_get in attrs_to_get:
            response = self.client.get_attr(attr_to_get)
            assert (
                200 <= response.status_code < 400
            ), f"Error retrieving {attrs_to_get}: {response.content}"

    # @unittest.skipIf(modlee_dev_available==False)
    def test_get_attrs_fail(self):
        """
        Trying to get random attributes should return none
        """
        attrs_to_fail = [f"fail_attr_{i}" for i in range(3)]
        for attr_to_fail in attrs_to_fail:
            response = self.client.get_attr(attr_to_fail)
            assert response is None, f"Should not have gotten attribute {attr_to_fail}"

    @unittest.skip
    def test_disconnected(self):
        """
        Fail to get a response from a dummy endpoint
        """
        dummy_ORIGIN = "http://9.9.9.9:9999"
        response = self.dummy_client.get()
        assert (
            response is None
        ), f"Should not have gotten a non-error status code from {dummy_ORIGIN}"

    @unittest.skipIf(
        modlee_dev_available == False,
        "modlee_dev not installed to test pypi-only env, skipping",
    )
    def test_callable(self):
        """
        Get callable objects (functions or classes)
        """
        callables_to_get = [
            "data_metafeatures.DataMetafeatures",
            "get_code_text",
            "get_code_text_for_model",
        ]
        for callable_to_get in callables_to_get:
            response = self.client.get_callable(callable_to_get)
            assert callable(
                response
            ), f"Could not retrieve callable object {callable_to_get}"

    @unittest.skipIf(
        modlee_dev_available == False,
        "modlee_dev not installed to test pypi-only env, skipping",
    )
    def test_fail_to_get_modules(self):
        """
        Cannot pickle modules so these should fail
        """
        modules_to_get = ["data_metafeatures", "utils", "rep"]
        for module_to_get in modules_to_get:
            response = self.client.get_attr(module_to_get)
            assert response is None, f"Should not have gotten module {module_to_get}"

    def test_get_script(self):
        """
        Get scripts as raw *.py files
        """
        scripts_to_get = ["data_metafeatures"]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.client.get_script(script_to_get)
            assert response, f"No response received (likely None)"
            assert (
                200 <= response.status_code < 400
            ), f"Could not get script {script_to_get}"

    def test_get_callable_from_script(self):
        """
        Get scripts as raw *.py files
        """
        scripts_to_get = ["data_metafeatures", "model_text_converter", "exp_loss_logger"]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.client.get_module(script_to_get)
            assert (
                response is not None
            ), f"Did not receive response, get_module({script_to_get}) returned None"
            # if modlee_dev_available:
            # try:
            exec(response, {}, locals())
            # except:
            # assert False, "Could not execute callable even though modlee_dev package available"

    @pytest.mark.deprecated
    def test_fail_dummy_gets_callable_from_script(self):
        """
        Get scripts as raw *.py files
        """
        scripts_to_get = ["data_metafeatures", "model_text_converter"]
        script_dict = {}
        for script_to_get in scripts_to_get:
            response = self.unauthorized_client.get_module(script_to_get)
            assert (
                response is None
            ), f"Unauthorized client should not have gotten {script_to_get}"

    @pytest.mark.deprecated
    def test_send_file(self):
        files_paths = [
            [
                os.path.join(os.path.dirname(__file__), "test_file.txt"),
                os.path.join(os.path.dirname(__file__), "test_file.txt"),
            ]
        ]

        for file_path in files_paths:
            [file_to_send, path_to_save] = file_path
            response = self.client.post_file(
                os.path.abspath(file_to_send), path_to_save
            )
            assert response, f"Could not post {file_path}"

    @pytest.mark.deprecated
    def test_post_run(self):

        for run_path in run_paths:
            response = self.client.post_run(run_path)
            assert response, f"Client {self.client.api_key} could not save {run_path}"

    @pytest.mark.deprecated
    def test_unauth_post_run(self):
        """ Unauthorized client should not be able to save runs 
        """
        for run_path in run_paths:
            response = self.unauthorized_client.post_run(run_path)
            assert (
                response is False
            ), f"Unauthorized client should not have saved {run_path}"

    # The 'server' marker excuses the client for known server-related issues
    @pytest.mark.server
    def test_post_run_as_json(self):

        for run_path in run_paths:
            response = self.client.post_run_as_json(run_path)
            assert response, f"Client {self.client.api_key} could not save {run_path}"

    @pytest.mark.server
    def test_unauth_post_run_as_json(self):
        """ Unauthorized client should not be able to save runs 
        """
        for run_path in run_paths:
            response = self.unauthorized_client.post_run(run_path)
            assert (
                response is False
            ), f"Unauthorized client should not have saved {run_path}"