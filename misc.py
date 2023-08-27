import sys
import importlib
from ast import literal_eval
import numpy as np
import math


#---------------------------------------------

def _discretize(n):

    if float(n)<0.1:
        ind = 2
        while str(n)[ind] == '0':
            ind += 1
        # print(ind)
        c = np.around(float(n),ind-1)
    elif float(n)<1.0:
        c = np.around(float(n),2)
    elif float(n)<10.:
        c = int(n)
    else:
        c = int(2**np.round(math.log(float(n))/math.log(2)))
    return c
        
def discretize(n: list[float, int]) -> list[float, int]:
    '''
    Discretize a list of inputs
    '''

    try:
    
        if type(n)==str:
            n = literal_eval(n)

        if type(n)==list:
            c = [_discretize(_n) for _n in n]
        elif type(n)==tuple:
            n = list(n)
            c = tuple([_discretize(_n) for _n in n])
        else:
            c = _discretize(n)
    except:
        c = n

    return c

def test_discretize():

  n = 0.234
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n = 0.00234
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n = 2.34
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n = 30143215
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n = [3.3,32144321,0.032]
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n = (1,23)
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n= 'test'
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))

  n=0.0005985885113477707
  n = str(n)
  print('input = {}, discretize(input)= {}'.format(n,discretize(n)))


def apply_discretize_to_summary(text,info):

    # text_split = [ [ p.split(key_val_seperator) for p in l.split(parameter_seperator)] for l in text.split(layer_seperator)] 
    # print(text_split)
    
    text_split = [ [ [ str(discretize(pp)) for pp in p.split(info.key_val_seperator)] for p in l.split(info.parameter_seperator)] for l in text.split(info.layer_seperator)] 
    # print(text_split)

    text_join = info.layer_seperator.join([ info.parameter_seperator.join([ info.key_val_seperator.join(p) for p in l]) for l in text_split])         
    # print(text_join)
        
    return text_join
        


