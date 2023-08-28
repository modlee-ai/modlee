#%%
import requests
import pickle
import inspect
flask_url = 'http://127.0.0.1:5000/'

get_api_obj = lambda url: pickle.loads(
    requests.get(url).content)

#%%
hw = get_api_obj(f"{flask_url}/hello_world")
# %%
hw()
#%%
inspect.getsource(hw)

# %%

class APIClient(object):
    def __init__(self,endpoint="",*args,**kwargs):
        self.endpoint = endpoint
    
    def _get_api_obj(self,url):
        response = requests.get(url)
        if response.status_code>400:
            print(
                # f"Responded with status code {response.status_code}\
                f"{response.status_code} error message: {response.content}"
                )
            return
        # breakpoint()
        response_content = response.content
        print(response_content)
        return pickle.loads(response_content)
        
    def get_from_route(self,route):
        return self._get_api_obj(
            f"{self.endpoint}/{route}"
        )
        
        
# %%
client = APIClient(endpoint="http://127.0.0.1:5000/")
# %%
hw = client.get_from_route('hello_world')

hw()

# %%
get_code_text = client.get_from_route(
    'modlee/get_code_text'
)

# %%
inspect.getsource(get_code_text)
# %%
DataStats = client.get_from_route(
    'modlee/data_stats.DataStats'
)
# %%
import modlee
dir(modlee)
print(modlee.data_stats.DataStats)
# %%
inspect.getsource(DataStats)

# %%
