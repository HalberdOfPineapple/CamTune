from .base_benchmark import BaseBenchmark
from .synthetic_functions import *
from .lunar_landing import Lunarlanding
from .mujoco import Swimmer, Hopper
from .mixed_synthetic import *

BENCHMARK_MAP = {
    'ackley2d': (AckleyBenchmark, {'dim': 2}),
    'ackley10d': (AckleyBenchmark, {'dim': 10}),
    'ackley20d': (AckleyBenchmark, {'dim': 20}),
    'ackley30d': (AckleyBenchmark, {'dim': 30}),
    'ackley50d': (AckleyBenchmark, {'dim': 50}),
    'ackley100d': (AckleyBenchmark, {'dim': 100}),

    'ackley_50d_eff20d': (EffectiveAckleyBenchmark, {'dim': 50, 'effective_dim': 20}),
    'ackley_100d_eff20d': (EffectiveAckleyBenchmark, {'dim': 100, 'effective_dim': 20}),
    'ackley20_200d': (EffectiveAckleyBenchmark, {'dim': 200, 'effective_dim': 20}),
    'ackley20_300d': (EffectiveAckleyBenchmark, {'dim': 300, 'effective_dim': 20}),
    'ackley2_300d': (EffectiveAckleyBenchmark, {'dim': 300, 'effective_dim': 2}),

    "shifted_ackley_50d": (ShiftedAckleyBenchmark, {'dim': 50, 'effective_dim': 50}),

    'rosenbrock10d': (RosenbrockBenchmark, {'dim': 10}),
    'rosenbrock20d': (RosenbrockBenchmark, {'dim': 20}),
    'rosenbrock30d': (RosenbrockBenchmark, {'dim': 30}),
    'rosenbrock50d': (RosenbrockBenchmark, {'dim': 50}),
    'rosenbrock100d': (RosenbrockBenchmark, {'dim': 100}),
    'rosenbrock2_200d': (EffectiveRosenbrockBenchmark, {'dim': 200, 'effective_dim': 2}),
    'rosenbrock20_200d': (EffectiveRosenbrockBenchmark, {'dim': 200, 'effective_dim': 20}),

    'levy10d': (LevyBenchmark, {'dim': 10}),
    'levy20d': (LevyBenchmark, {'dim': 20}),
    'levy30d': (LevyBenchmark, {'dim': 30}),
    'levy50d': (LevyBenchmark, {'dim': 50}),
    'levy100d': (LevyBenchmark, {'dim': 100}),

    'levy_50d_eff20d': (EffectiveLevyBenchmark, {'dim': 50, 'effective_dim': 20}),
    'levy2_200d': (EffectiveLevyBenchmark, {'dim': 200, 'effective_dim': 2}),
    'levy10_200d': (EffectiveLevyBenchmark, {'dim': 200, 'effective_dim': 10}),
    'levy20_200d': (EffectiveLevyBenchmark, {'dim': 200, 'effective_dim': 20}),

    'rastrigin10d': (RastriginBenchmark, {'dim': 10}),  # 'rastrigin10d': 'rastrigin10d',
    'rastrigin20d': (RastriginBenchmark, {'dim': 20}),
    'rastrigin30d': (RastriginBenchmark, {'dim': 30}),
    'rastrigin50d': (RastriginBenchmark, {'dim': 50}),
    'rastrigin100d': (RastriginBenchmark, {'dim': 100}),
    'rastrigin20_200d': (EffectiveRastriginBenchmark, {'dim': 200, 'effective_dim': 20}),

    'lunar12d': (Lunarlanding, {'dim': 12, 'lb': 0, 'ub': 12}),
    'swimmer16d': (Swimmer, {'dim': 16, 'lb': -1, 'ub': 1}),
    'hopper33d': (Hopper, {'dim': 33, 'lb': -1, 'ub': 1, 'negate': False}),

    'func2c': (Func2C, {}),
    'shifted_func2c': (ShiftedFunc2C, {}),
    'branin2': (BraninBenchmark, {'dim': 200, 'effective_dim': 2}),
    'branin2_300d': (BraninBenchmark, {'dim': 300, 'effective_dim': 2}),
    'branin2_500d': (BraninBenchmark, {'dim': 500, 'effective_dim': 2}),
    'hartmann6': (HartmannBenchmark, {'dim': 200, 'effective_dim': 6}),

    'lasso_hard': (LassoHardBenchmark, {})
}