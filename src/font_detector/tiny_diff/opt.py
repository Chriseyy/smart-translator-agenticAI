import numpy as np
import matplotlib.pyplot as plt
from tiny_diff.tensor import Tensor

def train(model, X, Y, loss_func, epochs=50, batch_size=2, lr=0.01):
    n = len(X)
    history = []
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    print(f"Training for {epochs} epochs with batch size {batch_size}, {n} samples and learning rate {lr}")

    for epoch in range(epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_indices = indices[i:i+batch_size]

            x_batch = Tensor(X[batch_indices])
            y_batch = Tensor(Y[batch_indices])

            # Calculate the prediction of the model
            preds = model(x_batch)

            # Calculate loss
            loss = loss_func(preds, y_batch)

            # Reset old gradients to 0 before calculating again
            model.zero_grad()
            # Calculate gradients
            loss.backward()

            # Update weights
            for param in model.parameters():
                param.data -= lr * param.grad

            epoch_loss += loss.data
            num_batches += 1

        # Calculate avg loss for the epoch
        avg_epoche_loss = epoch_loss / num_batches
        history.append(avg_epoche_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_epoche_loss:.6f}")

    return history

def plot_loss(history):
    plt.figure(figsize=(6,4))
    plt.plot(history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()