import torch
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from logging import Logger
from torch.utils.data import TensorDataset


def create_sequences(tokens: list[int], context_len: int) -> tuple[Tensor, Tensor]:
    """
    Creates input-target pairs for causal language modeling using a sliding window.

    For each position i in the token sequence, the function creates:
    - X = tokens[i : i + context_len]
    - Y = tokens[i + 1 : i + context_len + 1]

    Args:
        tokens: List of token indices representing the entire text.
        context_len: Length of the context window (number of tokens in each input sequence).

    Returns:
        A tuple (X, Y) where:
        - X has shape (num_sequences, context_len)
        - Y has shape (num_sequences, context_len)
    """
    X, Y = [], []

    for i in range(len(tokens) - context_len):
        X.append(tokens[i : i + context_len])
        Y.append(tokens[i + 1 : i + context_len + 1])

    return torch.tensor(X), torch.tensor(Y)


def get_dataset(*tensors: Tensor) -> TensorDataset:
    """
    Wraps tensors into a PyTorch TensorDataset for use with DataLoader.

    Args:
        *tensors: One or more tensors with the same first dimension
            (e.g., X and Y tensors).

    Returns:
        A TensorDataset containing the provided tensors.
    """
    return TensorDataset(*tensors)


def load_dataset_from_txt(path: Path, logger: Logger | None = None) -> str | None:
    """
    Safely loads the content of a text file into a single string.

    Args:
        path: Path object pointing to the text file to read.
        logger: Optional Logger instance for logging errors. If provided,
                any exceptions will be logged.

    Returns:
        The full content of the file as a string if successful, otherwise None.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        if logger:
            logger.error(f"File not found: {path}")
        return None
    except Exception as e:
        if logger:
            logger.exception(f"Failed to read file {path}: {e}")
        return None
