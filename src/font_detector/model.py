from tiny_diff.scalar.arch import MLP
from tiny_diff.scalar.node import relu

class FontSizeMLP(MLP):
    def __init__(self, input_dim: int):
        super().__init__(
            n_inputs=input_dim,
            layer_sizes=[64, 32, 1],
            activation=relu,
            name="FontSizeRegressor"
        )