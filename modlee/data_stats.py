# from modlee.api_client import ModleeAPIClient
from pathlib import Path

from modlee import modlee_client
# client = ModleeAPIClient(user_id='user1')
# globals().update(client.get_module('data_stats'))

# globals().update(client.get_module('data_stats'))
_module = modlee_client.get_module('data_stats')
module_available = False
if _module is not None:
    exec(_module,globals())
    module_available = True
