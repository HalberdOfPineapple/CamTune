import os
import json
import torch
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Dict

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .adapters import *
from camtune.utils.paths import BASE_DIR
LLAMA_SPACE_DIR = os.path.join(BASE_DIR, 'optimizer', 'llama_utils', 'spaces')

KNOB_TYPES = ['enum', 'integer', 'real', 'float']
VALID_ADAPTER_ALAIASES = ['none', 'tree', 'rembo', 'hesbo']


class ConfigSpaceGenerator:
    # Input by tpcc.all.llama.ini
    # definition -> knob dictionary specified in spaces/definitions/postgres-9.6.json
    # include = None
    # ignore = []
    # target_metric = throughput
    # adpater_alias = hesbo
    # le_low_dim = 16
    # bias_prob_sv = 0.2
    # quantization_factor = 10000
    # finalize_conf_func , unfinalized_conf_func => see spaces/common.py
    def __init__(
            self, 
            definition:dict=None, 
            include: Optional[List[str]]=None, 
            ignore: Optional[List[str]]=None, 
            target_metric=None,
            adapter_alias='none', 
            le_low_dim=None, 
            bias_prob_sv=None,
            quantization_factor=None, 
            finalize_conf_func=None, 
            unfinalize_conf_func=None,
            seed: int=1
        ):
        self.seed = seed
        if (ignore is not None) and (include is not None):
            raise ValueError("Define either `ignore_knobs' or `include_knobs'")
        assert isinstance(target_metric, str), 'Target metric should be of type string'
        assert adapter_alias in VALID_ADAPTER_ALAIASES, \
            f"Valid values for `adapter_alias' are: {VALID_ADAPTER_ALAIASES}"

        assert finalize_conf_func != None, 'Space should define "finalize_conf" function'
        self.finalize_conf_func = staticmethod(finalize_conf_func)
        assert unfinalize_conf_func != None, 'Space should define "unfinalize_conf" function'
        self.unfinalize_conf_func = staticmethod(unfinalize_conf_func)

        if bias_prob_sv is not None:
            bias_prob_sv = float(bias_prob_sv)
            assert 0 < bias_prob_sv < 1, 'Bias sampling prob should be in (0, 1) range'
        self._bias_prob_sv = bias_prob_sv

        if quantization_factor is not None:
            quantization_factor = int(quantization_factor)
            assert quantization_factor > 1, \
                'Quantization should be an integer value, larger than 1'
            assert (adapter_alias == 'none' and bias_prob_sv is None) or \
                    adapter_alias in ['rembo', 'hesbo']
        self._quantization_factor = quantization_factor

        # all_knobs = set([ d['name'] for d in  definition ])
        all_knobs = definition.keys()
        ignore = ignore if ignore is not None else []
        include_knobs = include if include is not None else all_knobs - set(ignore) # just all_knobs in TPCC-llama

        # self.knobs = [ info for info in definition if info['name'] in include_knobs ]
        # knob_info should include the knob_name as one of its fields
        self.knobs = []
        for knob_name, knob_info in definition.items():
            if knob_name in include_knobs:
                knob_info['name'] = knob_name
                self.knobs.append(knob_info)
        self.knobs_dict = { d['name']: d for d in self.knobs }

        self._target_metric = target_metric # to be used by the property
        self._adapter_alias = adapter_alias

        if adapter_alias in ['rembo', 'hesbo']:
            assert le_low_dim is not None, 'Linear embedding target dimensions not defined'
            self._le_low_dim = int(le_low_dim)
            assert self._le_low_dim < len(self.knobs), \
                'Linear embedding target dim should be less than original space'

    @property
    def target_metric(self):
        return self._target_metric
    
    def init_input_space(self, ignore_extra_knobs: Optional[List[str]]=None):
        self.input_variables = []
        self.lbs, self.ubs = [], []
        self.discrete_dims, self.continuous_dims = [], []
        self.enum_idx2val, self.enum_val2idx = {}, {}
        self.knob_to_idx = {}

        for info in self.knobs:
            name, knob_type = info['name'], info['type']
            if name in ignore_extra_knobs:
                continue
            
            if knob_type not in KNOB_TYPES:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')

            # Categorical variables
            if knob_type == 'enum':
                variable = CSH.CategoricalHyperparameter(
                                name=name,
                                choices=info["enum_values"],
                                default_value=info['default'])
                
                self.lbs.append(0)
                self.ubs.append(len(info["enum_values"]) - 1)

                var_idx = len(self.input_variables)
                self.discrete_dims.append(var_idx)
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

    def generate_input_space(self, seed: int, ignore_extra_knobs=None):
        ignore_extra_knobs = ignore_extra_knobs or [ ]

        self.init_input_space(ignore_extra_knobs)
        input_space = self.input_space

        self._input_space_adapter = None
        if self._adapter_alias == 'none':
            if self._bias_prob_sv is not None:
                # biased sampling
                self._input_space_adapter = PostgresBiasSampling(
                    input_space, seed, self._bias_prob_sv)
                return self._input_space_adapter.target

            if self._quantization_factor is not None:
                self._input_space_adapter = Quantization(
                    input_space, seed, self._quantization_factor)
                return self._input_space_adapter.target

            return input_space
        else:
            assert self._adapter_alias in ['rembo', 'hesbo']

            if self._bias_prob_sv is not None:
                # biased sampling
                input_space = PostgresBiasSampling(
                    input_space, seed, self._bias_prob_sv).target

            self._input_space_adapter = LinearEmbeddingConfigSpace.create(
                input_space, seed,
                method=self._adapter_alias,
                target_dim=self._le_low_dim,
                bias_prob_sv=self._bias_prob_sv,
                max_num_values=self._quantization_factor)
            return self._input_space_adapter.target

    def get_default_configuration(self):
        return self.input_space.get_default_configuration()

    def finalize_conf(self, conf, n_decimals=2):
        return self.finalize_conf_func.__func__(
                conf, self.knobs_dict, n_decimals=n_decimals)

    def unfinalize_conf(self, conf):
        return self.unfinalize_conf_func.__func__(conf, self.knobs_dict)

    def unproject_input_point(self, point):
        if self._input_space_adapter:
            return self._input_space_adapter.unproject_point(point)
        return point

    @classmethod
    def from_bounds(cls, bounds: torch.Tensor, discrete_dims: List[int], seed: int = 0):
        # Write a function to generate a ConfigSpace object from bounds
        # bounds: (2, D) tensor
        # discrete_dims: list of indices of discrete dimensions
        # seed: random seed
        raise NotImplementedError


    @classmethod
    def from_config(cls, spaces_config: dict, seed: int = 0):
        valid_config_fields = ['definition', 'include', 'ignore', 'target_metric',
                    'adapter_alias', 'le_low_dim', 'bias_prob_sv', 'quantization_factor']

        assert all(field in valid_config_fields for field in spaces_config), \
            'Configuration contains invalid fields: ' \
            f'{set(spaces_config.keys()) - set(valid_config_fields)}'
        assert 'definition' in spaces_config, 'Spaces section should contain "definition" key'
        assert 'include' in spaces_config or 'ignore' in spaces_config, \
                    'Spaces section should contain "include" or "ignore" key'
        assert not ('include' in spaces_config and 'ignore' in spaces_config), \
                    'Spaces section should not contain both "include" and "ignore" keys'
        assert 'target_metric' in spaces_config, \
                    'Spaces section should contain "target_metric" key'
        if 'le_low_dim' in spaces_config:
            assert spaces_config['adapter_alias'] in ['rembo', 'hesbo'], \
                'Linear embedding low dimension is only valid in REMBO & HesBO'

        # Read space definition from json filea
        # definition_fp = os.path.join(LLAMA_KNOB_DIR, f"{spaces_config['definition']}.json")
        # definition_fp = os.path.join(KNOB_DIR, f"{spaces_config['definition']}.json")
        definition_fp = spaces_config['definition']
        with open(definition_fp, 'r') as f:
            definition: dict = json.load(f)
        all_knobs_name = definition.keys()

        # Import designated module and utilize knobs
        include = 'include' in spaces_config
        module_name = spaces_config['include'] if include else spaces_config['ignore']
        relative_path = os.path.relpath(LLAMA_SPACE_DIR)  # Convert LLAMA_SPACE_DIR to a relative path
        import_path = module_name.replace('/', '.')
        module_path = f"{relative_path}.{import_path}".replace(os.sep, '.')
        module = import_module(module_path)
        
        knobs = getattr(module, 'KNOBS') 
        assert isinstance(knobs, list) and all([ k in all_knobs_name for k in knobs])
        finalize_conf_func = getattr(module, 'finalize_conf')
        unfinalize_conf_func = getattr(module, 'unfinalize_conf')

        return cls(
            definition=definition,
            include=knobs if include else None,
            ignore=knobs if not include else None,
            target_metric=spaces_config['target_metric'],
            adapter_alias=spaces_config.get('adapter_alias', 'none'),
            le_low_dim=spaces_config.get('le_low_dim', None),
            bias_prob_sv=spaces_config.get('bias_prob_sv', None),
            quantization_factor=spaces_config.get('quantization_factor', None),
            finalize_conf_func=finalize_conf_func,
            unfinalize_conf_func=unfinalize_conf_func,
            seed=seed
        )
