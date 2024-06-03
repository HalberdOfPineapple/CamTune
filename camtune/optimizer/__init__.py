from .base_optimizer import BaseOptimizer
from .random_optimizer import RandomOptimizer
from .constant_optimizer import ConstantOptimizer
from .turbo_optimizer import TuRBO
from .winter_mcts import WinterMCTS
from .baxus_optimizer import BAXUS
from .gp_sk_optimizer import GPROptimizer
from .mcts_ref import MCTSRefOptimizer
from .mcturbo import MCTuRBO
from .casmopolitan_ref import Casmopolitan
from .llama_optimizer import LlamaOptimizer
from .mcturboII import MCTuRBOII
from .bucket_optimizer import BucketOptimizer
from .spring_mcts import SpringMCTS
from .uniform import UniformOptimizer

OPTIMIZER_MAP = {
    "constant": ConstantOptimizer,
    "random": RandomOptimizer,
    "turbo": TuRBO,
    'mcts-winter': WinterMCTS,
    'mcts-ref': MCTSRefOptimizer,
    'baxus': BAXUS,
    'gp-sk': GPROptimizer,
    'mcturbo': MCTuRBO,
    'mcturboii': MCTuRBOII, 
    'casmo': Casmopolitan,
    'llama': LlamaOptimizer,
    'bucket': BucketOptimizer,
    'spring': SpringMCTS,
    'uniform': UniformOptimizer,
}


def build_optimizer(
    optimizer_type: str,
    **kwargs,
) -> BaseOptimizer:
    optimizer_type = optimizer_type.lower()
    if optimizer_type in OPTIMIZER_MAP:
        return OPTIMIZER_MAP[optimizer_type](**kwargs)
    else:
        raise ValueError(f"Undefined optimizer type: {optimizer_type}")