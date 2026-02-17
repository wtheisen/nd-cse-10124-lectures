import torch
import torch.nn.functional as F
import numpy as np
from embedding import EmbeddingLayer
from linear_layer import LinearLayer
from transformer import Transformer

class IrishChat:
    def __init__(self,  ctx_len: int = 1024, d_model: int = 128):
        self.padding_idx = 256
        self.ctx_len = ctx_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # TODO: Instantiate the components of the transformer
        self.layers = [
            EmbeddingLayer(512, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            Transformer(d_model, d_model, device=self.device),
            LinearLayer(d_model, 512, device=self.device),
        ]

    def forward(self, X, eval=False):
        """
        Args:
          x: (batch_size, seq_len) with integer token indices.
        Returns:
          logits: (batch_size, seq_len, vocab_size)
          outputs: (batch_size, seq_len, hidden_size) final hidden states over time.
        """
        # TODO: Calculate the output of the network
        for layer in self.layers:
            if isinstance(layer, Transformer):
                X = layer.forward(X, causal=True)
            else:
                X = layer.forward(X)

        return self.softmax(X)

    def backward(self, Y_hat, Y):
        # TODO: Calculate the gradient of the loss with respect to the input
        for layer in self.layers:
            if hasattr(layer, "zero_grads"):
                layer.zero_grads()

        mask = (Y.argmax(dim=-1) != self.padding_idx).unsqueeze(-1).float()
        normalizer = mask.sum().clamp_min(1.0)
        dA = (Y_hat - Y) * mask / normalizer

        for layer in reversed(self.layers):
            if isinstance(layer, Transformer):
                dA = layer.backward(dA, causal=True)
            else:
                dA = layer.backward(dA)

    def softmax(self, X):
        """
        Apply softmax to the input tensor.

        Args:
            X (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor with softmax applied, same shape as `X`.
        """
        return F.softmax(X, dim=-1)

    def cross_entropy(self, Y_hat, Y):
        """
        Compute the cross-entropy loss.

        Args:
            Y_hat (numpy.ndarray): Predicted probability matrix of shape (n_classes, m).
            Y (numpy.ndarray): One-hot encoded true labels of shape (n_classes, m).

        Returns:
            float: The average cross-entropy loss over all m examples.

        Notes:
            A small constant epsilon is added to Y_hat to avoid computing log(0).
        """

        # TODO: Calculate the cross-entropy loss
        log_probs = torch.log(Y_hat.clamp_min(1e-12))
        token_loss = -(Y * log_probs).sum(dim=2)  # (B, T)
        mask = (Y.argmax(dim=-1) != self.padding_idx).float()
        masked_loss = token_loss * mask
        normalizer = mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def get_accuracy(self, Y_hat, Y):
        """
        Compute the classification accuracy.

        Args:
            Y_hat (numpy.ndarray): Predicted probability matrix from the network, shape (n_classes, m).
            Y (numpy.ndarray): One-hot encoded true labels, shape (n_classes, m).

        Returns:
            float: Accuracy as a fraction between 0 and 1.
        """

        # TODO: Calculate the accuracy of the network
        # labels: Y (B, C) one-hot
        Y_idx = Y.argmax(dim=-1)                # (B,)
        preds = Y_hat.argmax(dim=-1)
        mask = (Y_idx != self.padding_idx).float()
        correct = (preds == Y_idx).float() * mask
        normalizer = mask.sum().clamp_min(1.0)
        return correct.sum() / normalizer

    def chat(self, prompt_tokens, max_new_tokens=200, temperature=1.0):
        """
        Autoregressive generation.

        Args:
            prompt_tokens: list of int token IDs (from tokenizer.encode)
            max_new_tokens: maximum number of tokens to generate
            temperature: 0 = greedy (always pick most likely),
                         >0 = sample (higher = more random)
        Returns:
            list of int token IDs (prompt + generated)
        """
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # Truncate to context window if the sequence is too long
            x = tokens[:, -self.ctx_len:]

            # Forward pass → probabilities (B, T, V)
            probs = self.forward(x)

            # We only care about the LAST position's prediction
            next_probs = probs[0, -1, :]        # (V,)

            # Pick the next token
            if temperature == 0:
                next_token = next_probs.argmax().item()
            else:
                # Scale probabilities by temperature and re-normalize
                scaled = next_probs.pow(1.0 / temperature)
                scaled = scaled / scaled.sum()
                next_token = torch.multinomial(scaled, 1).item()

            # Append to the sequence
            tokens = torch.cat([
                tokens,
                torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            ], dim=1)

            # Stop if we hit the EOS token (258 for Regex_Tokenizer)
            if next_token == 258:
                break

        return tokens[0].tolist()

    def train(self, train_loader, epochs=100, learning_rate=0.001, verbose=True):
        """
        Train the neural network using mini-batch gradient descent.

        Args:
            X (numpy.ndarray): Input data with shape (784, m), where each column is a flattened MNIST style image.
            Y (numpy.ndarray): One-hot encoded labels with shape (n_classes, m), where n_classes is 26
            epochs (int): Number of epochs for training.
            learning_rate (float): Learning rate for the parameter updates.
            batch_size (int, optional): Number of examples per mini-batch. Default is 32.
            verbose (bool, optional): If True, prints training progress every 500 epochs. Default is False.

        Returns:
            dict: A dictionary containing:
                - 'loss_history': List of loss values for each epoch.
                - 'accuracy_history': List of accuracy values for each epoch.

        Process:
            - Shuffles the dataset each epoch.
            - Processes data in mini-batches.
            - Performs a forward pass, backpropagation, and parameter updates for each mini-batch.
            - Computes the loss and accuracy for the entire dataset after each epoch.
        """
        loss_history = []
        accuracy_history = []
        
        for i in range(epochs):
            batch_losses = []
            batch_accuracies = []

            for X_batch, Y_batch in train_loader:
                # Forward propagation
                # TODO: Calculate the output of the network
                Y_hat_batch = self.forward(X_batch)

                # Calculate metrics for the whole epoch
                loss = self.cross_entropy(Y_hat_batch, Y_batch)
                accuracy = self.get_accuracy(Y_hat_batch, Y_batch)
                
                batch_losses.append(loss.item())
                batch_accuracies.append(accuracy.item())
                
                # Backward propagation
                # TODO: Calculate the gradients of the loss with respect to the input
                self.backward(Y_hat_batch, Y_batch)
                
                # Update parameters
                # TODO: Update the weights and biases of the layer using the learning rate
                for layer in self.layers:
                    layer.update(learning_rate)

            loss_history.append(np.mean(batch_losses))
            accuracy_history.append(np.mean(batch_accuracies))
            
            if verbose and i % (epochs // 10) == 0:
                print(f"Epoch {i+1}/{epochs}")
                print(f"loss: {loss_history[-1]:.5f}")
                print(f"accuracy: {accuracy_history[-1]:.5f}")
                print('Output test:')
                print("-" * 30)
        
        return {'loss_history': loss_history, 'accuracy_history': accuracy_history}