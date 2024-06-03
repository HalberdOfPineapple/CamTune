import json
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from camtune.utils import print_log, DTYPE, DEVICE

KNOB_TYPES = ['enum', 'integer', 'real', 'float']

class SearchSpace:
    def __init__(
          self, 
          knob_definition_path: str,
          is_kv_config: bool,
          include=None, 
          ignore=[],
          seed=1,
    ) -> None:
        self.include = None
        self.ignore = []
        self.seed = seed

        with open(knob_definition_path, 'r') as f:
            definitions = json.load(f)
        
        self.is_kv_config = is_kv_config
        if is_kv_config:
            self.all_knobs = set(definitions.keys())
            self.include_knobs = include if include is not None else self.all_knobs - set(ignore)

            self.knobs = [{'name':name, **info} for name, info in definitions.items() if name in self.include_knobs]
            self.knobs_dict = {d['name']: d for d in self.knobs}
        else:
            self.all_knobs = set([d['name'] for d in definitions])
            self.include_knobs = include if include is not None else self.all_knobs - set(ignore) 

            self.knobs = [info for info in definitions if info['name'] in self.include_knobs]
            self.knobs_dict = { d['name']: d for d in self.knobs}

        self._bounds = None
        self.init_input_space()
    
    def init_input_space(self):
        self.input_variables = []
        self.lbs, self.ubs = [], []
        self.discrete_dims, self.continuous_dims, self.cat_dims, self.integer_dims = [], [], [], []
        self.enum_idx2val, self.enum_val2idx = {}, {}
        self.knob_to_idx = {}

        for info in self.knobs:
            name, knob_type = info['name'], info['type']
            
            if knob_type not in KNOB_TYPES:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')

            # Categorical variables
            if knob_type == 'enum':
                variable = CSH.CategoricalHyperparameter(
                                name=name,
                                choices=info["enum_values"] if self.is_kv_config else info['choices'],
                                default_value=info['default'])
                
                self.lbs.append(0)
                self.ubs.append(len(info["enum_values"]) - 1)

                var_idx = len(self.input_variables)
                self.discrete_dims.append(var_idx)
                self.cat_dims.append(var_idx)
                
                self.enum_idx2val[var_idx] = {i: v for i, v in enumerate(info["enum_values"])}
                self.enum_val2idx[var_idx] = {v: i for i, v in enumerate(info["enum_values"])}

            # Discrete numerical variables
            elif knob_type == 'integer':
                variable = CSH.UniformIntegerHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
                # When inputting the value of knobs, we do not need to care about the unit
                # The knob value, shown in PostgreSQL, will automatically be divided by the unit and we only need to
                # care about checking whether the value is correctly applied.
                self.lbs.append(info['min'])
                self.ubs.append(info['max'])

                var_idx = len(self.input_variables)
                self.discrete_dims.append(var_idx)
                self.integer_dims.append(var_idx)
    
            # Continuous numerical variables
            elif knob_type == 'real' or knob_type == 'float':
                variable = CSH.UniformFloatHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
                self.lbs.append(info['min'])
                self.ubs.append(info['max'])

                var_idx = len(self.input_variables)
                self.continuous_dims.append(var_idx)
            else:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')
    
            self.knob_to_idx[name] = var_idx
            self.input_variables.append(variable)

        self.input_space = CS.ConfigurationSpace(name='input', seed=self.seed)
        self.input_space.add_hyperparameters(self.input_variables)
    
    def discrete_idx_to_value(self, knob_idx: int, num_val: int) -> int:
        knob_info: dict = self.knobs[knob_idx]
        if knob_info['type'] == 'enum':
            return self.enum_idx2val[knob_idx][num_val]
        elif knob_info['type'] == 'integer':
            return num_val
        else:
            raise NotImplementedError(f'Knob type of "{knob_info["type"]}" does not need to be mapped.')
    
    @property
    def bounds(self) -> torch.Tensor:
        if self._bounds is None:
            self._bounds = torch.tensor([self.lbs, self.ubs], device=DEVICE, dtype=DTYPE) # (2, D)
        return self._bounds
    
    def get_default_conf_tensor(self):
        default_conf: dict = dict(self.input_space.get_default_configuration())
        # print_log(f'[SearchSpace] Default configuration: {default_conf}', print_msg=True)

        default_conf_tensor = torch.zeros(1, len(default_conf), device=DEVICE, dtype=DTYPE)
        for k, v in default_conf.items():
            knob_idx = self.knob_to_idx[k]
            if self.knobs[knob_idx]['type'] == 'enum':
                v = self.enum_val2idx[knob_idx][v]
            default_conf_tensor[0, knob_idx] = v
        # print_log(f'[SearchSpace] Default configuration tensor: {default_conf_tensor}', print_msg= True)
        return default_conf_tensor
        
