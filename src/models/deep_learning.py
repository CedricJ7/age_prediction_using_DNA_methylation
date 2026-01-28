"""
Deep Learning models for age prediction using PyTorch.

Implements DeepMAge-inspired architecture for DNA methylation age prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class DeepMAge(nn.Module):
    """
    DeepMAge-inspired neural network for epigenetic age prediction.

    Architecture:
        Input -> Linear(512) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(1) -> Output

    Parameters
    ----------
    input_size : int
        Number of input features (CpG sites)
    hidden_size : int, default=512
        Number of neurons in hidden layer
    dropout : float, default=0.3
        Dropout probability for regularization
    """

    def __init__(self, input_size, hidden_size=512, dropout=0.3):
        super(DeepMAge, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x).squeeze(-1)


class DeepMAgeRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for DeepMAge PyTorch model.

    Parameters
    ----------
    hidden_size : int, default=512
        Number of neurons in hidden layer
    dropout : float, default=0.3
        Dropout probability
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=100
        Number of training epochs
    early_stopping_patience : int, default=10
        Patience for early stopping (epochs without improvement)
    device : str, optional
        Device to use ('cuda' or 'cpu'). Auto-detected if None.
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print training progress
    """

    def __init__(
        self,
        hidden_size=512,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        device=None,
        random_state=42,
        verbose=True
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.scaler_ = None
        self.training_history_ = []
        self.best_epoch_ = None
        self.best_loss_ = None

    def _get_device(self):
        """Get the device to use for computation."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y, eval_set=None):
        """
        Fit the DeepMAge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        eval_set : list of tuples, optional
            Validation set in format [(X_val, y_val)]

        Returns
        -------
        self : object
            Fitted estimator
        """
        self._set_seed()
        device = self._get_device()

        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Initialize model
        input_size = X.shape[1]
        self.model_ = DeepMAge(
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout
        ).to(device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Validation data
        X_val_tensor = None
        y_val_tensor = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            X_val_scaled = self.scaler_.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)

        # Training loop
        self.training_history_ = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()

            # Create batches
            n_samples = len(X_tensor)
            indices = torch.randperm(n_samples)

            epoch_train_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model_(X_batch)
                loss = criterion(predictions, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                n_batches += 1

            epoch_train_loss /= n_batches

            # Validation phase
            epoch_val_loss = None
            if X_val_tensor is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_predictions = self.model_(X_val_tensor)
                    epoch_val_loss = criterion(val_predictions, y_val_tensor).item()

            # Store history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss
            }
            if epoch_val_loss is not None:
                history_entry['val_loss'] = epoch_val_loss
            self.training_history_.append(history_entry)

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {epoch_train_loss:.4f}"
                if epoch_val_loss is not None:
                    msg += f" - Val Loss: {epoch_val_loss:.4f}"
                print(msg)

            # Early stopping
            if epoch_val_loss is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    self.best_epoch_ = epoch + 1
                    self.best_loss_ = epoch_val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                        print(f"Best epoch: {self.best_epoch_} with val loss: {self.best_loss_:.4f}")
                    # Restore best model
                    self.model_.load_state_dict({k: v.to(device) for k, v in self.best_model_state_.items()})
                    break

        return self

    def predict(self, X):
        """
        Predict using the trained DeepMAge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted ages
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        device = self._get_device()

        # Convert to numpy and scale
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # Predict
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
            predictions = predictions.cpu().numpy()

        return predictions

    @property
    def best_iteration(self):
        """Return the best iteration (for compatibility with XGBoost interface)."""
        return self.best_epoch_ if self.best_epoch_ is not None else self.epochs


def create_deepmage_model(config):
    """
    Create a DeepMAge regressor with the given configuration.

    Parameters
    ----------
    config : ModelConfig
        Configuration object containing model parameters

    Returns
    -------
    model : DeepMAgeRegressor
        Configured DeepMAge model
    """
    return DeepMAgeRegressor(
        hidden_size=config.deepmage_hidden_size,
        dropout=config.deepmage_dropout,
        learning_rate=config.deepmage_learning_rate,
        batch_size=config.deepmage_batch_size,
        epochs=config.deepmage_epochs,
        early_stopping_patience=config.deepmage_early_stopping_patience,
        random_state=config.deepmage_random_state,
        verbose=True
    )
