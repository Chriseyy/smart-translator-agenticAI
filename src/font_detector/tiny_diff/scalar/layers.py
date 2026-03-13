from tiny_diff.scalar.node import Node


import random
from math import sqrt
from typing import List, Callable, Any

class Neuron:
    """
    A single neuron with a list of input weights, a bias, and an activation function.
    """
    def __init__(self, n_inputs: int, init_limit: float, activation: Callable[[Any], Any], name: str = "Neuron") -> None:
        """
        Initialize a neuron.

        Args:
            n_inputs (int): Number of input connections.
            init_limit (float): Limit for uniform weight initialization.
            activation (callable): Activation function.
            name (str): Name of the neuron (used for naming Node parameters).
        """
        # Weights initialized uniformly in [-init_limit, init_limit]
        self.w: List[Node] = [
            Node(random.uniform(-init_limit, init_limit), name=f"{name}.w{i}")
            for i in range(n_inputs)
        ]
        # Bias initialized to zero
        self.b: Node = Node(0.0, name=f"{name}.b")
        self.activation: Callable[[Any], Any] = activation

    def __call__(self, x: List[Node]) -> Node:
        """
        Compute the output of the neuron for input x.

        Args:
            x (list of Node): Input nodes.

        Returns:
            Node: The activated output.
        """
        # Start with bias
        z: Node = self.b
        # Weighted sum of inputs
        for wi, xi in zip(self.w, x):
            z = z + wi * xi
        # Apply activation function
        return self.activation(z)

    def parameters(self) -> List[Node]:
        """
        Return all parameters (weights and bias) of the neuron.

        Returns:
            list of Node: All trainable parameters.
        """
        return [self.b] + self.w


class FCL:
    """
    Fully Connected Layer (dense layer) consisting of multiple neurons.
    """
    def __init__(self, n_inputs: int, n_outputs: int, activation: Callable[[Any], Any], name: str = "FCL") -> None:
        """
        Initialize the fully connected layer.

        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of neurons/output features.
            activation (callable): Activation function for all neurons.
            name (str): Name of the layer.
        """
        # Xavier/Glorot initialization limit
        init_limit = sqrt(6 / (n_inputs + n_outputs))
        print(f"Init Limit: {init_limit}")

        # Create neurons
        self.neurons: List[Neuron] = [
            Neuron(n_inputs, init_limit, activation=activation, name=f"{name}.n{j}")
            for j in range(n_outputs)
        ]
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.activation: Callable[[Any], Any] = activation
        self.name: str = name

    def __call__(self, x: List[Node]) -> List[Node]:
        """
        Compute the outputs of all neurons in the layer.

        Args:
            x (list of Node): Input nodes.

        Returns:
            list of Node: Outputs from each neuron.
        """
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> List[Node]:
        """
        Return all parameters (weights and biases) of all neurons.

        Returns:
            list of Node: Flattened list of all trainable parameters.
        """
        params: List[Node] = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self) -> str:
        """
        String representation showing layer details.
        """
        return (f"{self.name}(in_features={self.n_inputs}, "
                f"out_features={self.n_outputs}, activation={str(self.activation)})")
