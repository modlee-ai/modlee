import os
import requests
# import pickle
# import dill as pickle
import cloudpickle as pickle
import functools

LOCAL_ENDPOINT = "http://127.0.0.1:7070"
REMOTE_ENDPOINT = "http://modlee.pythonanywhere.com"


class ModleeAPIClient(object):
    def __init__(self, endpoint=LOCAL_ENDPOINT, api_key=None, *args, **kwargs):
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
        try:
            with open(file,'r') as _file:
                file_text = _file.read()
        except:
            print(f"No file {file}")
        res = self.post(
            route="postfile",
            data={
                'file_text':file_text,
                'filepath':filepath
                },
            )
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
                # self.post_file()
            