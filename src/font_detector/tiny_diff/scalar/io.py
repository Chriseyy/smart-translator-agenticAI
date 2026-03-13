import json
from typing import Any


def save_model_weights(model: Any, file_path: str) -> None:
    """
    Save model parameters to a JSON file.

    The model's parameters are assumed to be objects with `name` and `value` attributes.
    Each parameter is stored as a key-value pair {name: value}.

    Args:
        model: A model object with a `parameters()` method returning parameter objects.
        file_path (str): Path to save the JSON file.
    """
    # Get all parameters from the model
    params = model.parameters()

    # Convert parameters to a dictionary {name: value}
    param_dict = {p.name: float(p.value) for p in params}

    # Write the dictionary to a JSON file
    with open(file_path, "w") as f:
        json.dump(param_dict, f, indent=2)

    print(f"Model saved to {file_path}")


def load_model_weights(model: Any, file_path: str) -> None:
    """
    Load model parameters from a JSON file.

    The JSON should contain a dictionary {name: value}. Parameters in the model
    are matched by name. Any parameters in the model not found in the file are skipped.

    Args:
        model: A model object with a `parameters()` method returning parameter objects.
        file_path (str): Path to the JSON file containing saved weights.
    """
    # Load parameter dictionary from the JSON file
    with open(file_path, "r") as f:
        param_dict = json.load(f)

    # Track how many parameters are loaded or skipped
    loaded, skipped = 0, 0

    # Iterate over all parameters in the model
    for p in model.parameters():
        if p.name in param_dict:
            # Set parameter value from the JSON
            p.value = float(param_dict[p.name])
            loaded += 1
        else:
            # Parameter in model not found in file
            skipped += 1
            print(f"Warning: parameter {p.name} not found in file")

    print(f"Model loaded from {file_path} ({loaded} parameters restored, {skipped} skipped)")
