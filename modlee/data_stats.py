from modlee.api_client import ModleeAPIClient

client = ModleeAPIClient()
DataStats = client.get_object('data_stats.DataStats')
