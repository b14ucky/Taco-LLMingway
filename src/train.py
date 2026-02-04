import torch
from tqdm import tqdm
import torch.nn as nn
from logging import Logger
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime


class Trainer:
    """
    Trainer class for training a PyTorch model with logging, checkpointing,
    and optional loss plotting.
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

    def train(
        self,
        n_epochs: int,
        final_model_path: Path,
        checkpoint_dir: Path,
        plot_loss: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_interval: int = 10,
    ) -> None:
        """
        Runs training for a given number of epochs, with optional checkpointing
        and loss visualization.

        Args:
            n_epochs: Number of epochs to train the model.
            final_model_path: Path where the final model checkpoint will be saved.
            checkpoint_dir: Directory where intermediate checkpoints will be saved.
            plot_loss: Whether to plot the loss curve after training. Default is True.
            checkpoint_path: Optional path to a checkpoint to resume training from.
            checkpoint_interval: Number of epochs between checkpoint saves.

        Returns:
            None
        """
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        loss_data = []
        for epoch in range(n_epochs):
            total_loss = 0

            for X, y in tqdm(self.dataloader, desc=f"epoch {epoch + 1}/{n_epochs}"):
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)

                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info(
                f"epoch: {epoch + 1} | loss: {total_loss / len(self.dataloader):.5f}"
            )
            loss_data.append(total_loss / len(self.dataloader))

            if epoch + 1 % checkpoint_interval == 0 and epoch != n_epochs:
                filename = f"{datetime.now().strftime('%Y%m%d')}_{epoch + 1}_{final_model_path.stem}.pth"
                self._save_model(checkpoint_dir / filename)

        self._save_model(final_model_path)

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

    def _save_model(self, path: Path) -> None:
        """
        Saves the model and optimizer state to a checkpoint file.

        This method will automatically create the parent directory of `path`
        if it does not already exist.

        Args:
            path: Filesystem path where the model checkpoint should be saved.

        Returns:
            None
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "optimizer": self.optimizer.state_dict(),
        }
        if self.device == "cuda":
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()

        torch.save(data, path)
        self.logger.info(f"Model state saved in: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """
        Loads model and optimizer state from a previously saved checkpoint.

        The checkpoint is loaded using `map_location=self.device` to ensure
        compatibility when moving between GPU and CPU environments.

        Args:
            path: Path to a checkpoint file created with `_save_model`.

        Returns:
            None
        """
        if not path.exists():
            self.logger.error(f"File not found: {path}")
            return None

        data = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(data["optimizer"])

        if self.device == "cuda":
            self.model.module.load_state_dict(data["model"])
        else:
            self.model.load_state_dict(data["model"])

        self.logger.info(f"Checkpoint: {path} successfully loaded")
