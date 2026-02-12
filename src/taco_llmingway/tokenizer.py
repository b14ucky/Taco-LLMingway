from typing import Literal
from pathlib import Path
import json
from logging import Logger


class Tokenizer:
    """
    Simple tokenizer that builds a vocabulary from a text dataset.
    """

    def __init__(
        self,
        dataset: str,
        tokenization_mode: Literal["character", "word"] = "character",
    ) -> None:
        """
        Initializes the Tokenizer

        Args:
            dataset: Text data from which the vocabulary will be built.
            tokenization_mode: Determines whether tokens are characters or words.
        """
        self._tokenization_mode = tokenization_mode
        self._vocab_encode = self._create_vocab(dataset)
        self._vocab_decode = {v: k for k, v in self._vocab_encode.items()}

    def __len__(self) -> int:
        """
        Returns the size of the vocabulary as integer.

        Returns:
            Size of the vocabulary (int).
        """
        return len(self._vocab_encode)

    def encode(self, text: str) -> list[int]:
        """
        Converts text into a list of token indices.

        Args:
            text: Input string to be tokenized and encoded.

        Returns:
            A list of integer token indices corresponding to the input text.
        """
        if self._tokenization_mode == "character":
            return [
                self._vocab_encode.get(char, self._vocab_encode["<unk>"])
                for char in text
            ]
        if self._tokenization_mode == "word":
            return [
                self._vocab_encode.get(word, self._vocab_encode["<unk>"])
                for word in text.split(" ")
            ]

        raise ValueError(f"Unknown tokenization_mode: {self._tokenization_mode}")

    def decode(self, indices: list[int]) -> str:
        """
        Converts a list of token indices back into a string.

        Args:
            indices: List of integer token indices.

        Returns:
            A decoded string reconstructed from the provided indices.
        """
        if self._tokenization_mode == "character":
            return "".join(
                [self._vocab_decode.get(index, "<unk>") for index in indices]
            )
        if self._tokenization_mode == "word":
            return " ".join(
                [self._vocab_decode.get(index, "<unk>") for index in indices]
            )

        raise ValueError(f"Unknown tokenization_mode: {self._tokenization_mode}")

    def _create_vocab(
        self,
        dataset: str,
    ) -> dict[str, int]:
        """
        Creates a vocabulary mapping tokens to integer indices.

        Args:
            dataset: The text to tokenize and build the vocabulary from.

        Returns:
            A dictionary mapping each token to a unique integer index.
        """
        if self._tokenization_mode == "character":
            vocab = {
                token: index
                for index, token in enumerate(sorted(set(list(dataset))))
            }
        elif self._tokenization_mode == "word":
            vocab = {
                token: index
                for index, token in enumerate(sorted(set(list(dataset.split(" ")))))
            }
        else:
            raise ValueError(f"Unknown tokenization_mode: {self._tokenization_mode}")

        vocab["<unk>"] = len(vocab)

        return vocab

    def save(self, path: Path, logger: Logger | None = None) -> None:
        """
        Saves the tokenizer configuration to a JSON file.

        The saved file contains the tokenization mode and the vocabulary
        mapping required to reconstruct the tokenizer instance.

        Args:
            path: Path to the output JSON file.
            logger: Optional logger used to log save status information.

        Returns:
            None
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tokenization_mode": self._tokenization_mode,
            "vocab": self._vocab_encode,
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

        if logger:
            logger.info(f"Tokenizer saved in {path}")

    @classmethod
    def load(cls, path: Path, logger: Logger | None = None) -> "Tokenizer":
        """
        Loads a tokenizer instance from a JSON file.

        The file must contain the tokenization mode and vocabulary mapping
        previously saved using the `save` method.

        Args:
            path: Path to the tokenizer JSON file.
            logger: Optional logger used to log load status information.

        Returns:
            A reconstructed Tokenizer instance.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If required fields are missing in the JSON file.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            tokenizer = cls.__new__(cls)
            tokenizer._tokenization_mode = data["tokenization_mode"]
            tokenizer._vocab_encode = data["vocab"]
            tokenizer._vocab_decode = {
                v: k for k, v in tokenizer._vocab_encode.items()
            }

            if logger:
                logger.info(f"Succesfully loaded tokenizer from {path}")

            return tokenizer

        except FileNotFoundError as e:
            if logger:
                logger.exception(f"File {path} not found")

            raise e
