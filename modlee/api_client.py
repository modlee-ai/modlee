import os
import requests
# import pickle
# import dill as pickle
import cloudpickle as pickle
import functools

LOCAL_ENDPOINT = "http://127.0.0.1:7070"
# REMOTE_ENDPOINT = "http://modlee.pythonanywhere.com"
REMOTE_ENDPOINT = "http://ec2-3-84-155-233.compute-1.amazonaws.com:7070"

class ModleeAPIClient(object):
    """
    A client for making requests to the API
    """
    def __init__(self, endpoint=LOCAL_ENDPOINT, api_key=None, *args, **kwargs):
        """
        Args:
            endpoint (_type_, optional): The server endpoint. Defaults to LOCAL_ENDPOINT.
            api_key (_type_, optional): The user's API key. Defaults to None.
        """
        
        if api_key=='local':
            endpoint = LOCAL_ENDPOINT
        elif api_key is not None:
            endpoint = REMOTE_ENDPOINT
        self.endpoint = endpoint
        self.get_object = self.get_callable
        self.get_function = self.get_callable
        self.timeout = 3
        self.api_key = api_key

    @property
    def available(self):
        """
        Ping the server

        Returns:
            _type_: _description_
        """
        ret = self.get()
        if ret is not None:
            return 200 <= ret.status_code < 400
        else:
            return False

    def post(self, route="", *args, **kwargs):
        return self._request(route=route, method="post", *args, **kwargs)

    def get(self, route="", *args, **kwargs):
        return self._request(route=route, method="get", *args, **kwargs)

    def _request(self, route="", method="get", *args, **kwargs):
        """
        Send request to the endpoint

        Args:
            route (str, optional): The route to send request. Defaults to "".
            method (str, optional): The request method in lowercase (e.g. get, post). Defaults to "get".

        Returns:
            _type_: response from the server or None if an error (response.status_code>=400)
        """
        req_url = f"{self.endpoint}/{route}"
        # if method=="get":
        kwargs.update(dict(timeout=self.timeout))
            
        kwargs.update({
            'auth':(self.api_key,self.api_key),
            'headers':{
                'User-Agent': 'Mozilla/5.0',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"}
            })
        
        try:
            ret = getattr(requests, method)(
                req_url,
                *args, 
                **kwargs
            )
            if ret.status_code >= 400:
                if ret.status_code != 404:
                    pass
                ret = None
        except requests.Timeout as e:
            ret = None
        except requests.ConnectionError as e:
            ret = None
        return ret

    def login(self, user_id=''):
        """
        Log the user in
        """
        return self.post(route=f"login",data={'user_id':user_id})

    def get_attr(self, route=""):
        return self.get(f"modlee/{route}")

    def get_callable(self, route=""):
        ret = self.get_attr(route)
        if ret is not None:
            ret = pickle.loads(ret.content)
        return ret

    def get_script(self, route=""):
        return self.get(f"modleescript/{route}")

    def get_module(self, route=""):
        
        ret = self.get_script(route)
        if ret is not None:
            ret = ret.content
        return ret

    def post_file(self, file, filepath):
        """
        Save a file (as text) to save on the server

        Args:
            file (_type_): The local file
            filepath (_type_): The relative path on the server at which to save the file

        Returns:
            _type_: Server request response
        """
        try:
            with open(file,'rb') as _file:
                # file_text = _file.read()
                res = self.post(
                    route="postfile",
                    data={
                        # 'file_text':file_text,
                        'filepath':filepath
                        },
                    files={
                        'file':_file
                        # filepath:file
                    }
                    )
                return res
        
            with open(file,'rb') as _file:
                res = self.post(
                    route="postfile",
                    data={
                        
                        'filepath':filepath,
                    }
                )
        except:
            print(f"Could not access file {file}")
            return None
        return res
        
    def save_run(self,run_dir):
        """
        Save a run given a directory,
        returns True if all successful
        or False if some failures

        Args:
            run_dir (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # early return if API key is not set
        # if self.api_key is None:
        #     return False
        ignore_files = [
            'model.pth',
            '.npy',
            '.DS_Store',
            '__pycache__',
        ]
        
        error_files = []
        
        def skip_file(rel_filepath):
            """
            Skip file if ignore_files in filepath
            """
            for ignore_file in ignore_files:
                if ignore_file in rel_filepath:
                    return True
            return False
                
        run_id = os.path.basename(run_dir)
        for dirs_files in os.walk(run_dir):
            base_dir,_,files = dirs_files
            for file in files:
                filepath = os.path.join(base_dir,file)
                rel_filepath = filepath.split(run_id)[-1]

                if skip_file(rel_filepath=rel_filepath):
                    continue
                server_filepath = '/'.join([
                    self.api_key,
                    run_id,
                    rel_filepath,                    
                ])

                res = self.post_file(filepath, server_filepath)
                if res is None:
                    error_files.append(rel_filepath)
        if len(error_files)>0:
            return False
        return True