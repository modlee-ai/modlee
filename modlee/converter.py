import inspect

import torch
import torchsummary
import onnx2torch
import onnx_graphsurgeon as gs
import onnx

# TODO - cache models for "longer" pipelines
# e.g. torch2code do not need to convert to onnx every time
class Converter(object):
    def __init__(self):
        pass
    
    def torch2onnx(self,torch_model):
        # TODO - generalize this input_dummy
        # This is a placeholder for ResNet-based models
        input_dummy = torch.randn([10,3,300,300])
        onnx_path = f'./test_model.onnx'
        torch.onnx.export(
            torch_model,
            input_dummy,
            onnx_path)
        return onnx.load(onnx_path)
        
    def onnx_path2torch(self,onnx_path):
        return onnx2torch.convert(onnx_path)
    
    def onnx2torch(self,onnx_model):
        return onnx2torch.convert(onnx_model)
        
    def torch2code(self,torch_model):
        return self.get_model_code(
            self.onnx2torch(
            self.torch2onnx(
            torch_model)))
        
    def code2torch(self,torch_code):
        exec(torch_code,locals())
        return Model()
        
    def torch2torch(self,torch_model):
        return self.code2torch(
            self.torch2code(
            torch_model))
        
    def save_torch(self,torch_model,filepath):
        with open(filepath,'w') as _file:
            _file.write(self.torch2code(torch_model))
        
    def get_inner_string(self,s,_start,_end):
        s = s[s.find(_start)+len(_start):]
        s = s[:s.rfind(_end)]
        # valid attributes should just be one string
        if len(s.split())==1:
            return s
        else:
            return None

    def get_attr_name(self,s):
        attr_name = self.get_inner_string(s,
            _start = 'getattr(self, "',
            _end = '")',
        )
        if attr_name is None:
            attr_name = self.get_inner_string(s,
                _start = 'self.',
                _end = '(',
            )
        return attr_name
            
        
    def get_model_attr_on_line(self,model,line):
        attr_name = self.get_attr_name(line)
        if attr_name:
            try:
                return {attr_name:getattr(model,attr_name)}
            except:
                return {}
        
    def get_model_attrs_in_forward(self,model):
        fwd_source = inspect.getsource(model.forward)
        fwd_lines = fwd_source.split('\n')
        model_attrs = {}
        for fwd_line in fwd_lines[1:]:
            model_attr = self.get_model_attr_on_line(model, fwd_line)
            if model_attr:
                # model_attrs.append(model_attr)
                model_attrs.update(model_attr)
        return model_attrs

    def get_params_for_attr(self,model_attr):
        attrs_to_skip = ['bias']
        attr_kwargs = dict(inspect.signature(model_attr.__init__).parameters)
        attr_params = {}
        for attr_key,attr_val in attr_kwargs.items():
            if attr_key in attrs_to_skip: continue
            # if attr_val.default==inspect._empty:
            # if True:
            if hasattr(model_attr,attr_key):
                attr_params.update({attr_key:getattr(model_attr,attr_key)})        
        return attr_params

    def get_type_string(self,obj):
        return self.get_inner_string(
            str(obj.__class__),
            _start="<class '",
            _end="'>"
        )

    def get_init(self,model):
        model_attrs = self.get_model_attrs_in_forward(model)        

        init_attrs = []

        for model_attr_name,model_attr in model_attrs.items():
            attr_params = self.get_params_for_attr(model_attr)
            init_attrs.append((model_attr_name,
                self.get_type_string(model_attr), attr_params))
        # init_code = 'def __init__(self):\n'
        spacer = "    "
        spacer = "    "
        init_lines = [
'',
f'{spacer}def __init__(self):',
f'{spacer*2}super().__init__()',
        ]
        for init_attr in init_attrs:
            init_line = f"{spacer*2}setattr(self,'{init_attr[0]}', {init_attr[1]}(**{init_attr[2]}))"
            init_lines.append(init_line)
        init_code = '\n'.join(init_lines)
        return init_code

    def get_forward(self,model):
        spacer = "    "
        fwd_lines = inspect.getsource(model.forward).split('\n')
        for i,fwd_line in enumerate(fwd_lines[1:]):
            fwd_lines[i+1] = f"{spacer}{fwd_line}"
        return '\n'.join(fwd_lines)

    def get_model_code(self,model):
        
        model_init_code = self.get_init(model)
        model_fwd_code = self.get_forward(model)
        spacer = "    "
        model_code = f"""
import torch, onnx2torch

class Model(torch.nn.Module):
{spacer}{model_init_code}

{spacer}{model_fwd_code}
"""
        return model_code
    
    
"""
Test line:
python3 -c 'from modlee.converter import Converter as Con; import torchvision.models as models; Con().torch2code(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))'
"""