import os
import logging
import importlib
from .client import ModleeClient


class ModleeAPIConfig:
    """
    Singleton class to manage API configuration and client initialization for Modlee.
    Ensures that only one instance of the configuration exists and handles API key validation and client setup.
    """

    _instance = None

    def __new__(cls):
        """
        Creates a new instance of the class if it doesn't already exist.
        Initializes the API key from the environment variable and sets up the Modlee client.

        :return: The singleton instance of ModleeAPIConfig.
        """
        if cls._instance is None:
            cls._instance = super(ModleeAPIConfig, cls).__new__(cls)
            cls._instance.api_key = os.environ.get("MODLEE_API_KEY", None)
            cls._instance.client = ModleeClient(api_key=cls._instance.api_key)
        return cls._instance

    def set_api_key(self, api_key):
        """
        Sets the API key for the Modlee configuration and updates the client.
        Reloads the modules to ensure they are using the updated client.

        :param api_key: The new API key to be set.
        """
        self.api_key = api_key
        os.environ["MODLEE_API_KEY"] = self.api_key
        self.client = ModleeClient(api_key=self.api_key)
        self.reload_modules()

    def reload_modules(self):
        """
        Reloads the Modlee-related modules to ensure they are using the updated client.
        """
        import modlee.data_metafeatures as data_metafeatures
        import modlee.model_text_converter as model_text_converter

        importlib.reload(data_metafeatures)
        importlib.reload(model_text_converter)
        if model_text_converter.module_available:
            from modlee.model_text_converter import (
                get_code_text,
                get_code_text_for_model,
            )

    def get_client(self):
        """
        Returns the Modlee client after ensuring that the API key is correctly set.

        :return: The Modlee client.
        """
        self.ensure_api_key()
        return self.client

    def ensure_api_key(self):
        """
        Ensures that the API key is set and matches the client's API key.
        Logs warnings if the API key is not set or if there is a mismatch.

        :return: True if the API key is set and matches the client's API key, False otherwise.
        """
        if self.api_key is None:
            logging.warning("API key is not set. Functionality will be limited.")
            return False

        if self.api_key != self.client.api_key:
            logging.warning(
                "Stale API key identified, Please ensure the API key used in code and one stored as the environment variable are the same."
            )
            return False

        return True
