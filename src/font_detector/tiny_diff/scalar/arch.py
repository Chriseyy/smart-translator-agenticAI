from tiny_diff.scalar.layers import FCL
from tiny_diff.scalar.node import tanh, identity

from typing import List, Callable, Any, Optional

class Sequence:
    """
    A container for layers that applies them sequentially.
    Acts like a simple neural network model.
    """
    def __init__(self, layers: List[Any], name: Optional[str] = None) -> None:
        """
        Initialize the sequence.

        Args:
            layers (list): List of layer objects.
            name (str, optional): Name for the sequence.
        """
        self.layers: List[Any] = layers
        self.name: Optional[str] = name

    def __call__(self, x: Any) -> Any:
        """
        Make the sequence callable. Applies each layer in order.

        Args:
            x: Input to the network (scalar, vector, or batch).

        Returns:
            Output after applying all layers.
        """
        out = x
        for layer in self.layers:
            out = layer(out)  # apply current layer
        return out

    def parameters(self) -> List[Any]:
        """
        Collect parameters from all layers that have a `parameters` method.

        Returns:
            list: All parameters of all layers.
        """
        params: List[Any] = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        """
        String representation of the sequence showing each layer.
        """
        return f"{self.name}(\n  " + "\n  ".join(repr(l) for l in self.layers) + "\n)"


class MLP(Sequence):
    """
    Multi-Layer Perceptron (MLP) implemented as a sequence of fully connected layers.
    """
    def __init__(
        self,
        n_inputs: int,
        layer_sizes: List[int],
        activation: Callable[[Any], Any] = tanh,
        name: str = "MLP"
    ) -> None:
        """
        Initialize an MLP.
        Example:
        mlp = MLP(10, [8, 4, 1], activation=relu):
        An MLP with layers (n_input, n_output): (10:8), (8:4), (4:1).

        Args:
            n_inputs (int): Number of input features.
            layer_sizes (list of int): Number of neurons in each hidden/output layer.
            activation (callable, optional): Activation function for hidden layers (last layer: always identity).
            name (str, optional): Name for the MLP.
        """
        # Prepend input size to layer sizes
        layer_sizes = [n_inputs] + layer_sizes
        layers: List[Any] = []

        for i in range(len(layer_sizes) - 1):
            # Use the specified activation for all layers except the last (linear output)
            act = activation if i < len(layer_sizes) - 2 else identity

            # Create a fully connected layer
            layer = FCL(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=act,
                name=f"{name}.L{i}"
            )
            layers.append(layer)

        # Initialize the Sequence superclass with the created layers
        super().__init__(layers, name)
