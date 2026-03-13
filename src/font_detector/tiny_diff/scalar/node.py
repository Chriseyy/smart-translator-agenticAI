import math
from typing import Callable, Tuple, List, Optional, Union

class Node:
    """
    A scalar value with automatic differentiation (autodiff) support.
    Each Node tracks its parents and the local backward function.
    """
    def __init__(
        self,
        value: float,
        parents: Tuple["Node", ...] = (),
        backward_fn: Optional[Callable[[float], None]] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            value (float): Scalar value of this node.
            parents (tuple of Node, optional): Parent nodes contributing to this Node.
            backward_fn (callable, optional): Function to propagate gradients to parents.
            name (str, optional): Optional name for debugging.
        """
        self.value: float = float(value)
        self.grad: float = 0.0  # accumulated gradient
        # flatten parents into a tuple
        self.parents: Tuple[Node, ...] = tuple(
            p for parent in parents for p in (parent if isinstance(parent, (list, tuple)) else [parent])
        )
        self._backward = backward_fn
        self.name: Optional[str] = name

    def __repr__(self) -> str:
        return f"Node(value={self.value:.4f}, grad={self.grad:.4f}, name={self.name})"

    def backward(self) -> None:
        """
        Compute gradients of this Node with respect to all upstream nodes.
        Implements reverse-mode autodiff.
        """
        self.grad = 1.0  # seed gradient
        topo: List[Node] = []
        visited: set = set()

        def build_topo(node: "Node") -> None:
            if node in visited:
                return
            visited.add(node)
            for p in node.parents:
                build_topo(p)
            topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            if node._backward:
                node._backward(node.grad)


def _to_node(x: Union[Node, float, int]) -> Node:
    """Convert a scalar or Node to a Node object."""
    if isinstance(x, Node):
        return x
    elif isinstance(x, (int, float)):
        return Node(x)
    else:
        raise TypeError(f"Cannot convert {type(x)} to Node")


# -------------------- Basic operations -------------------- #

def add(a: Node, b: Node) -> Node:
    val = a.value + b.value
    def _back(grad: float) -> None:
        a.grad += grad
        b.grad += grad
    return Node(val, (a, b), _back)

def sub(a: Node, b: Node) -> Node:
    val = a.value - b.value
    def _back(grad: float) -> None:
        a.grad += grad
        b.grad -= grad
    return Node(val, (a, b), _back)

def neg(a: Node) -> Node:
    val = -a.value
    def _back(grad: float) -> None:
        a.grad += -grad
    return Node(val, (a,), _back, name=f"(-{a.name})")

def mul(a: Node, b: Node) -> Node:
    val = a.value * b.value
    def _back(grad: float) -> None:
        a.grad += grad * b.value
        b.grad += grad * a.value
    return Node(val, (a, b), _back)

def div(a: Node, b: Node) -> Node:
    val = a.value / b.value
    def _back(grad: float) -> None:
        a.grad += grad / b.value
        b.grad -= grad * a.value / (b.value ** 2)
    return Node(val, (a, b), _back)

def pow(a: Node, n: float) -> Node:
    val = a.value ** n
    def _back(grad: float) -> None:
        a.grad += grad * n * (a.value ** (n - 1))
    return Node(val, (a,), _back)

# -------------------- Activation functions -------------------- #

def tanh(a: Node) -> Node:
    val = math.tanh(a.value)
    def _back(grad: float) -> None:
        a.grad += grad * (1 - val**2)
    return Node(val, (a,), _back)

def relu(a: Node) -> Node:
    val = max(0, a.value)
    def _back(grad: float) -> None:
        if a.value > 0:
            a.grad += grad

    return Node(val, (a,), _back)

def identity(a: Node) -> Node:
    val = a.value
    def _back(grad: float) -> None:
        a.grad += grad
    return Node(val, (a,), _back)

def exp(a: Node) -> Node:
    val = math.exp(a.value)
    def _back(grad: float) -> None:
        a.grad += grad * val
    return Node(val, (a,), _back)

def sin(a: Node) -> Node:
    val = math.sin(a.value)
    def _back(grad: float) -> None:
        a.grad += grad * math.cos(a.value)
    return Node(val, (a,), _back)

def sigmoid(a: Node) -> Node:
    val = 1 / (1 + math.exp(-a.value))
    def _back(grad: float) -> None:
        a.grad += grad * val * (1 - val)
    return Node(val, (a,), _back)

def log(a: Node) -> Node:
    val = math.log(a.value)
    def _back(grad: float) -> None:
        a.grad += grad / a.value
    return Node(val, (a,), _back)


# -------------------- Softmax -------------------- #

def softmax(logits: List[Node]) -> List[Node]:
    """
    Compute softmax over a list of Nodes.
    Returns a list of Nodes with backward properly defined.
    """
    exps = [exp(l) for l in logits]
    s = sum(exps)
    outputs = [Node(e.value / s.value) for e in exps]

    def _back(grad_outputs: List[float]) -> None:
        # grad_outputs: dL/ds_i
        soft_vals = [o.value for o in outputs]
        for j, l in enumerate(logits):
            grad_j = 0.0
            for i, g in enumerate(grad_outputs):
                if i == j:
                    grad_j += g * soft_vals[i] * (1 - soft_vals[j])
                else:
                    grad_j -= g * soft_vals[i] * soft_vals[j]
            l.grad += grad_j

    # Wrap each output Node to include backward
    wrapped_outputs: List[Node] = []
    for i, o in enumerate(outputs):
        def _back_i(grad: float, i=i) -> None:
            grads = [0.0 for _ in outputs]
            grads[i] = grad
            _back(grads)
        wrapped_outputs.append(Node(o.value, (logits,), _back_i))

    return wrapped_outputs


# -------------------- Operator overloading -------------------- #

Node.__add__ = lambda self, other: add(self, _to_node(other))
Node.__radd__ = lambda self, other: add(_to_node(other), self)

Node.__sub__ = lambda self, other: sub(self, _to_node(other))
Node.__rsub__ = lambda self, other: sub(_to_node(other), self)

Node.__mul__ = lambda self, other: mul(self, _to_node(other))
Node.__rmul__ = lambda self, other: mul(_to_node(other), self)

Node.__truediv__ = lambda self, other: div(self, _to_node(other))
Node.__rtruediv__ = lambda self, other: div(_to_node(other), self)

Node.__pow__ = lambda self, n: pow(self, n)
Node.__neg__ = lambda self: neg(self)
