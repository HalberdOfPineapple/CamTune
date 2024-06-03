import os
import logging
import torch
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict, Any, Callable
from ConfigSpace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

from .space import ConfigSpaceGenerator
from .adapters import LHDesignWithBiasedSampling

from camtune.utils import print_log, get_result_dir, get_expr_name, get_log_idx
# from camtune.utils.paths_llama import get_smac_output_dir


# def get_smac_optimizer_from_bounds(
#     input_space:
# )

def get_smac_optimizer(
        llama_config: dict, 
        spaces: ConfigSpaceGenerator, 
        obj_func: Callable,
        ignore_knobs=None, 
    ):
    # Generate input (i.e. knobs) and output (i.e. perf metric) space
    input_space = spaces.generate_input_space(
        llama_config['seed'], 
        ignore_extra_knobs=ignore_knobs
    )
    return get_smac_optimizer_(llama_config, input_space, obj_func, ignore_knobs=ignore_knobs)

def get_smac_optimizer_(
    llama_config: dict, 
    input_space: ConfigurationSpace, 
    obj_func: Callable,
    ignore_knobs=None, 
):
    # print_log(f'[SMAC_Optimizer] input space:\n{input_space}')
    middle_fix = '' if get_log_idx() == 0 else f'_{get_log_idx()}'
    smac_output_dir = os.path.join(get_result_dir(), f'{get_expr_name()}{middle_fix}_smac')

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": llama_config['num_evals'],
        "cs": input_space,
        "deterministic": "true",
        "always_race_default": "false",
        "limit_resources": "false", # disables pynisher, which allows for shared state,
        "output_dir": smac_output_dir,
    })
    # Latin Hypercube design, with 10 iters
    init_rand_samples = int(llama_config['optimizer'].get('init_rand_samples', 10))
    initial_design = LHDesignWithBiasedSampling
    initial_design_kwargs = {
        "init_budget": init_rand_samples,
        "max_config_fracs": 1,
    }

    # Get RF params from config
    rand_percentage = float(llama_config['optimizer']['rand_percentage'])
    assert 0 <= rand_percentage <= 1, 'Optimizer rand optimizer must be between 0 and 1'
    n_estimators = int(llama_config['optimizer']['n_estimators'])

    #  how often to evaluate a random sample
    random_configuration_chooser_kwargs = {
        'prob': rand_percentage,
    }
    
    model_type = 'rf' if 'model_type' not in llama_config['optimizer'] else llama_config['optimizer']['model_type']
    assert model_type in ['rf', 'gp', 'mkbo'], 'Model type %s not supported' % model_type

    if model_type == 'rf':
        # RF model params -- similar to MLOS ones
        model_kwargs = {
            'num_trees': n_estimators,
            'log_y': False,         # no log scale
            'ratio_features': 1,    #
            'min_samples_split': 2, # min number of samples to perform a split
            'min_samples_leaf': 3,  # min number of smaples on a leaf node
            'max_depth': 2**20,     # max depth of tree
        }
        optimizer = SMAC4HPO(
            scenario=scenario,
            tae_runner=obj_func,
            rng=llama_config['seed'],
            model_kwargs=model_kwargs,
            initial_design=initial_design,
            initial_design_kwargs=initial_design_kwargs,
            random_configuration_chooser_kwargs=random_configuration_chooser_kwargs,
        )

    elif model_type == 'gp':
        optimizer = SMAC4BB(
            model_type='gp',
            scenario=scenario,
            tae_runner=obj_func,
            rng=llama_config['seed'],
            initial_design=initial_design,
            initial_design_kwargs=initial_design_kwargs,
            random_configuration_chooser_kwargs=random_configuration_chooser_kwargs,
        )

    # logger.info(optimizer)
    print_log(f'[SMAC_Optimizer] Optimizer: {optimizer}', print_msg=True)
    return optimizer
