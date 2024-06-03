from .local_control import MCTSLocalControl
from .node import Node
from .turbo_component import TuRBO

OPTIMIZER_MAP = {
    'turbo': TuRBO,
}