import lightning.pytorch as pl
from modlee import modlee_client

_module = modlee_client.get_module('loss_sweeper')
module_available = False
if _module is not None:
    exec(_module,globals())
    module_available = True
