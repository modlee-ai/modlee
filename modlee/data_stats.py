from modlee.api_client import ModleeAPIClient
from pathlib import Path

client = ModleeAPIClient()
# globals().update(client.get_module('data_stats'))

# globals().update(client.get_module('data_stats'))
_module = client.get_module('data_stats')
module_available = False
if _module is not None:
    exec(_module,globals())
    module_available = True
