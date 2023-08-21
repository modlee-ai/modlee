from modlee.api_client import ModleeAPIClient
from pathlib import Path

client = ModleeAPIClient()
# globals().update(client.get_module('data_stats'))

# globals().update(client.get_module('data_stats'))
exec(client.get_module('data_stats'),globals())