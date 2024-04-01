""" 
Modlee client for modlee server.
"""
import os
import requests

# import pickle
# import dill as pickle
import cloudpickle as pickle
import functools

LOCAL_ORIGIN = "http://127.0.0.1:7070"
# REMOTE_ORIGIN = "http://modlee.pythonanywhere.com"
# REMOTE_ORIGIN = "http://ec2-3-84-155-233.compute-1.amazonaws.com:7070"
from modlee.config import SERVER_ORIGIN as REMOTE_ORIGIN
import json

class ModleeClient(object):
    """
    A client for making requests to the server.
    """

    def __init__(self, origin=LOCAL_ORIGIN, api_key=None, *args, **kwargs):
        """
        ModleeClient constructor.
        
        :param origin: The server origin (scheme://hostname:port).
        :param api_key: The user's API key for authenticating functionality to the server.
        """

        if api_key == "kF4dN7mP9qW2sT8v":
            origin = LOCAL_ORIGIN
        elif api_key is not None:
            origin = REMOTE_ORIGIN
        self.origin = origin
        self.get_object = self.get_callable
        self.get_function = self.get_callable
        self.timeout = 3
        self.api_key = api_key

    @property
    def available(self):
        """
        Check if the server is available.

        :return: Whether the server is available.
        """
        ret = self.get()
        if ret is not None:
            return 200 <= ret.status_code < 400
        else:
            return False

    def post(self, path="", *args, **kwargs):
        """
        Post a request.

        :param path: The URL path, defaults to "".
        :return: The response.
        """
        return self._request(path=path, method="post", *args, **kwargs)

    def get(self, path="", *args, **kwargs):
        """
        Get a request.

        :param path: The URL path, defaults to "".
        :return: The response.
        """
        return self._request(path=path, method="get", *args, **kwargs)

    def _request(self, path="", method="get", *args, **kwargs):
        """
        Send request to the origin

        :param path: The URL path, defaults to "".
        :param method: The request method, defaults to "get".
        :return: The response.
        """
        req_url = f"{self.origin}/{path}"

        # kwargs.update(dict(timeout=self.timeout))
        kwargs['timeout'] = kwargs.get('timeout',self.timeout)

        # Set headers only if not already defined 
        kwargs_headers = kwargs.get("headers", {
                    "User-Agent": "Mozilla/5.0",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*",
                    "X-API-KEY": self.api_key,
                    },
                )
        kwargs_headers.update({
            "X-API-KEY": self.api_key,
        })
        kwargs.update(
            {
                "auth": (self.api_key, self.api_key),
                "headers": kwargs_headers,
            }
        )
        # breakpoint()
 
        try:
            # if method=='post': breakpoint()
            ret = getattr(requests, method)(req_url, *args, **kwargs)
            # breakpoint()
            if ret.status_code >= 400:
                if ret.status_code != 404:
                    pass
                ret = None
        except requests.Timeout as e:
            ret = None
        except requests.ConnectionError as e:
            ret = None
        return ret

    def login(self, api_key=""):
        """
        Log the user in.
        
        :param api_key: The user's ID
        """
        return self.post(path=f"login", data={"user_id": api_key})

    def get_attr(self, path=""):
        """
        Get an attribute from the server.

        :param path: The server-side path of the attribute to get, defaults to "".
        :return: The server attribute.
        """
        return self.get(f"modlee/{path}")

    def get_callable(self, path=""):
        """
        Get a callable function from the server.
        
        :param path: The server-side path of the callable to get, defaults to "".
        :return: The callable, or None if not retrievable.
        """
        ret = self.get_attr(path)
        if ret is not None:
            ret = pickle.loads(ret.content)
        return ret

    def get_script(self, path=""):
        """
        Get a script from the server.

        :param path: The server-side path of the script to get, defaults to "".
        :return: The script as text, or None if not retrievable.
        """
        return self.get(f"modleescript/{path}")

    def get_module(self, path=""):
        """
        Get a module from the server.

        :param path: the server-side path of the module to get, defaults to "".
        :return: The module, or None if not retrievable.
        """

        ret = self.get_script(path)
        if ret is not None:
            ret = ret.content
        return ret

    def post_file(self, file, filepath):
        """
        Post a file (as text) to save on the server.

        :param file: The local file to send.
        :param filepath: The server-side relative path for the file.
        :return: Response if successful, None if failed.
        """
        try:
            with open(file, "rb") as _file:
                # file_text = _file.read()
                res = self.post(
                    path="postfile",
                    data={
                        # 'file_text':file_text,
                        "filepath": filepath
                    },
                    files={
                        "file": _file
                        # filepath:file
                    },
                )
                return res

            with open(file, "rb") as _file:
                res = self.post(path="postfile", data={"filepath": filepath})
        except:
            print(f"Could not access file {file}")
            return None

    def post_run(self, run_path):
        """
        Post a local experiment run to save on the server.

        :param run_path: The path of the run to save.
        :return: Whether saving all files was successful. Partial failures still return False.
        """
        ignore_files = ["model.pth", ".npy", ".DS_Store", "__pycache__"]

        error_files = []

        def skip_file(rel_filepath):
            """
            Skip file if ignore_files in filepath
            """
            for ignore_file in ignore_files:
                if ignore_file in rel_filepath:
                    return True
            return False

        run_id = os.path.basename(run_path)
        # Check that there are items in the directory
        if not os.path.exists(run_path) or len(os.listdir(run_path)) < 1:
            return False

        for dirs_files in os.walk(run_path):
            base_dir, _, files = dirs_files
            for file in files:
                filepath = os.path.join(base_dir, file)
                rel_filepath = filepath.split(run_id)[-1]

                if skip_file(rel_filepath=rel_filepath):
                    continue
                server_filepath = "/".join([self.api_key, run_id, rel_filepath])

                res = self.post_file(filepath, server_filepath)
                if res is None:
                    error_files.append(rel_filepath)
        if len(error_files) > 0:
            return False
        return True


    def post_run_as_json(self, run_path):
        """
        Convert the specified mlruns' directory structure and files to JSON format, 
        excluding files that match patterns in ignore_files. Tracks files that could not be processed.

        :param run_path: The path of the run to convert to JSON.
        :return: A dictionary representing the directory structure and files in JSON format, 
                 or None if the directory is empty or doesn't exist.
        """
        ignore_files = ["model.pth", ".npy", ".DS_Store", "__pycache__"]
        ignore_exts = ["ckpt"]
        error_files = []

        def skip_file(file_name):
            """
            Skip file if it matches any pattern in ignore_files.

            :param file_name: The name of the file to check.
            :return: True if the file should be skipped, False otherwise.
            """
            for ignore_file in ignore_files:
                if ignore_file in file_name:
                    return True
                elif os.path.splitext(file_name)[-1] in ignore_exts:
                    return True
            return False

        def dir_to_json(path):
            """
            Recursively convert a directory structure to a JSON-compatible dictionary.

            :param path: The directory path to convert.
            :return: A dictionary representing the directory's structure and contents.
            """
            result = {}
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    result[item] = dir_to_json(item_path)  # Recurse into subdirectories
                elif not skip_file(item):
                    try:
                        with open(item_path, 'r') as file:
                            result[item] = json.load(file)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        try:
                            with open(item_path, 'r') as file:
                                result[item] = file.read()  # Read file content as plain text
                        except Exception as e:
                            print(f"Error reading file {item_path}: {e}")
                            error_files.append(item_path)
                            result[item] = f"Error reading file: {e}"
            return result

        # Check that there are items in the directory
        if not os.path.exists(run_path) or len(os.listdir(run_path)) < 1:
            return False
        json_data = dir_to_json(run_path)

        if len(error_files) > 0:
            print(f"Error processing the following files: {error_files}")
            return False
        
          # Define the temporary file path
        temp_file_path = 'logs.json'
        with open(temp_file_path, 'w') as tmp_file:
            json.dump(json_data, tmp_file, indent=4)

        # Post the file
        run_id = os.path.basename(run_path)
        
        
        server_filepath = "/".join([self.api_key, run_id]) #, 'logs.json'])

        res = self.post_file(temp_file_path, server_filepath)                
        os.remove(temp_file_path)
        if res is None:
            return False
        else:
            return True
