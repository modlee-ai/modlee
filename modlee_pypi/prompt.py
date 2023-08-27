from ast import literal_eval
import math
import numpy as np

from modlee.misc import discretize

def create_parameter_text(key,key_val_seperator,val):
    if type(val)==str:
        return "{}{}{}".format(key,key_val_seperator,val)
    else:
        return "{}{}{}".format(key,key_val_seperator,val)


def get_prompt(output_dim,dataset_dims,difficulty_metrics,min_acc,max_tp,max_lat,task,info):

    # global key_val_seperator,parameter_seperator
    
    prompt_dict = {'output_dim':output_dim,
                   'dataset_dims':dataset_dims,
                   'min_acc':float(min_acc),
                   'max_tp':max_tp,
                   'max_lat':float(max_lat),
                   'task':task,
                  }
    
    prompt_dict.update(difficulty_metrics)
        
    p = [ create_parameter_text(key,info.key_val_seperator,val) for key,val in prompt_dict.items()]
    
    return info.parameter_seperator.join(p)

def get_prompt_details(prompt_text,info):
    
    parameter_seperator,key_val_seperator = info.parameter_seperator,info.key_val_seperator
    
    details = {p.split(key_val_seperator)[0]:p.split(key_val_seperator)[1] for p in prompt_text.split(parameter_seperator)}

    for key,val in details.items():
        try:
            details[key]=literal_eval(val)
        except:
            continue
    
    return details


def generate_adjusted_prompts(prompt,tuner_summary,info,num_gen=20):
    
    prompt_split = [ [ str(discretize(pp)) for pp in p.split(info.key_val_seperator)] for p in prompt.split(info.parameter_seperator)] 
    # print(prompt_split)    
    prompt_split_dict = {p[0]:p[1] for p in prompt_split}
    # print(prompt_split_dict)

    tuner_split = [ [ str(discretize(pp)) for pp in p.split(info.key_val_seperator)] for p in tuner_summary.split(info.parameter_seperator)] 
    # print(tuner_split)
    tuner_split_dict = {t[0]:t[1] for t in tuner_split}
    # print(tuner_split_dict)
    
    adjusted_prompts = []
    
    acc = float(tuner_split_dict['accuracy'])
    max_tp = float(tuner_split_dict['total_parameters'])
    max_tp_power = int(np.ceil(math.log(float(max_tp))/math.log(2)))
    
    for _ in range(num_gen):
        
        prompt_split_dict['min_acc']=np.around(np.random.uniform(0.0,acc),2)
        prompt_split_dict['max_tp']=int(2**np.random.randint(max_tp_power,40))
        adjusted_prompt = info.parameter_seperator.join([info.key_val_seperator.join([p[0],str(prompt_split_dict[p[0]])]) for p in prompt_split])
        adjusted_prompts.append(adjusted_prompt)
        
    return adjusted_prompts
