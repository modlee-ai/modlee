import requests
# import pickle
# import dill as pickle
import cloudpickle as pickle
import functools

DEFAULT_ENDPOINT = "http://127.0.0.1:5000"


class ModleeAPIClient(object):
    def __init__(self, endpoint=DEFAULT_ENDPOINT, api_key=None, *args, **kwargs):
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
        if method=="get":
            kwargs.update(dict(timeout=self.timeout))
            
        kwargs.update({'auth':(self.api_key,self.api_key)})
        try:
            ret = getattr(requests, method)(
                req_url,
                *args, 
                **kwargs
            )
            if ret.status_code >= 400:
                ret = None
        except requests.Timeout as e:
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
            # _locals = {}
            # exec(ret, {}, _locals)
            # ret = _locals
        return ret
