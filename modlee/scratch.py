



#-------------------------------
# Manually Set Core "text" elements

data_stats = 'data-stats'

#set by user for prompt
target_req = 'target_req'

model = 'model-code-text'

def test_eval():
  return None
evaluation_def = test_eval

performance = 'performance'

#-------------------------------
# MLOps to set Core Elements

data_stats,model,evaluation_def,performance = modlee.extract_from_mlflow('path_to_exp_and_run',data_stats=data_stats)

#set by user for prompt
target_req = 'target_req'

#-------------------------------

def check_api_key(key):
  return True

def share_exp_with_api(exp):
  pass

#-------------------------------

def convert_data_stats(data_stats):
  #discretize numbers
  #organize systematically
  #create single text element with correct spacing
  return 'data-stats'

def convert_model(model):
  #discretize numbers
  #organize systematically
  #create single text element with correct spacing
  return 'model'

def convert_evaluation_def(model):
  #discretize numbers
  #organize systematically
  #create single text element with correct spacing
  return 'model'

def convert_performance(model):
  #discretize numbers
  #organize systematically
  #create single text element with correct spacing
  return 'model'


element_text_seperators = '<???>'

class Modlee:

  def __init__(self,api_key='0203040201'):
    
    self.api_key = api_key
    self.valid_key = check_api_key(api_key)#Bool
    
  def document_experiment(self,data_stats,model,evaluation_def,performance):
    # data_stats = dict
    # model = Pytorch Model
    # evaluation_def = self contained python def
    # performance = dict : {'acc':96}
    #--------------------------------------

    #--- convert to text forms
    data_stats = convert_data_stats(data_stats)
    #note: we will inject target requirements later in prompt engineering
    model = convert_model(model)
    evaluation_def = convert_evaluation_def(evaluation_def)
    performance = convert_performance(performance)

    exp = element_text_seperators.join([data_stats,model,evaluation_def,performance])
    share_exp_with_api(self.api_key,exp)


  def document_mlflow_experiment(self,data_stats,path_exp_run='path/to/exp/run/'):

    model,evaluation_def,performance = extract_from_mlflow('path_to_exp_and_run')#loop through paths, grab correct objects

    exp = self.document_experiment(data_stats,model,evaluation_def,performance)
    share_exp_with_api(self.api_key,exp)

  def get_model_suggestion():
    pass




