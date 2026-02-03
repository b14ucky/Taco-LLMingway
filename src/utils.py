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


def load_dataset_from_txt(path: Path, logger: Logger) -> str | None:
    """
    Safely loads the content of a text file or a directory containing text files
    into a single string. If a directory is provided, all `.txt` files are read
    recursively and concatenated.

    Args:
        path: Path object pointing to a text file or a directory containing
            text files.
        logger: Logger instance used to log errors and debug information.

    Returns:
        A single string containing the concatenated content of all read text
        files if successful, or None if the path does not exist or an error
        occurs while reading any file.
    """
    if not path.exists():
        logger.error(f"File/directory not found: {path.absolute()}")
        return None

    if path.is_dir():
        content = ""

        file_list = list(path.rglob("*.txt"))
        logger.debug(
            f"Found {len(file_list)} files: {", ".join(str(file) for file in file_list)}"
        )
        for file in file_list:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content += f.read()
            except Exception as e:
                logger.exception(f"Unable to open file {file}: {e}")
                return None

        return content

    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.exception(f"Failed to read file {path}: {e}")
        return None
