import torch
from tqdm import tqdm
import torch.nn as nn
from logging import Logger
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class for training a PyTorch model with logging and optional loss plotting.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: Logger,
        lr: float,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            dataloader: DataLoader providing (X, y) batches.
            logger: Logger instance for logging info and debug messages.
            lr: Learning rate for the AdamW optimizer.
        """
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Device: {self.device}")
        self.dataloader = dataloader
        self.logger.debug(f"DataLoader len: {len(self.dataloader)}")

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def train(self, n_epochs: int, vocab_size: int, plot_loss: bool = True) -> None:
        """
        Runs training for a given number of epochs.

        Args:
            n_epochs: Number of epochs to train the model.
            vocab_size: Size of the vocabulary (number of classes).
            plot_loss: Whether to plot the loss curve after training. Default is True.

        Returns:
            None
        """
        loss_data = []
        for i in range(n_epochs):
            total_loss = 0

            for X, y in tqdm(self.dataloader, desc=f"epoch {i + 1}/{n_epochs}"):
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)

                loss = self.criterion(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info(
                f"epoch: {i + 1} | loss: {total_loss / len(self.dataloader) :.5f}"
            )
            loss_data.append(total_loss / len(self.dataloader))

        if plot_loss:
            self._plot_loss(loss_data)

    def _plot_loss(self, data: list[float]) -> None:
        """
        Plots the training loss curve.

        Args:
            data: List of loss values (one per epoch).

        Returns:
            None
        """
        plt.plot(data)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Train loss")
        plt.show()
