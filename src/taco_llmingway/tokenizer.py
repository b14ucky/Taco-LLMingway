import json
import re
from enum import StrEnum
from logging import Logger
from pathlib import Path
from typing import Literal


class TokenType(StrEnum):
    """
    Enumeration of token categories used during regex-based tokenization.

    Attributes:
        SPECIAL: Matches user-defined special tokens.
        NORMAL: Matches standard text (characters not classified as special).
    """

    SPECIAL = "SPECIAL"
    NORMAL = "NORMAL"


class Tokenizer:
    """
    Simple tokenizer that builds a vocabulary from a text dataset.
    """

    def __init__(
        self,
        dataset: str,
        special_tokens: list[str] | None = None,
        tokenization_mode: Literal["character", "word"] = "character",
    ) -> None:
        """
        Initializes the Tokenizer.

        Args:
            dataset: Text data from which the vocabulary will be built.
            special_tokens: Optional list of special tokens that should be treated
                as atomic units during tokenization.
            tokenization_mode: Determines whether tokens are characters or words.
        """
        self._tokenization_mode = tokenization_mode
        self._special_tokens = special_tokens
        if self._special_tokens is not None:
            self._compiled_regex = self._compile_regex(
                "|".join(re.escape(token) for token in self._special_tokens)
            )
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
            if self._special_tokens is not None:
                indices: list[int] = []

                for mo in self._compiled_regex.finditer(text):
                    kind = mo.lastgroup
                    value = mo.group()
                    if kind == TokenType.SPECIAL:
                        indices.append(
                            self._vocab_encode.get(value, self._vocab_encode["<unk>"])
                        )
                    elif kind == TokenType.NORMAL:
                        indices.extend(
                            self._vocab_encode.get(char, self._vocab_encode["<unk>"])
                            for char in value
                        )

                return indices

            else:
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

    def _create_vocab(self, dataset: str) -> dict[str, int]:
        """
        Creates a vocabulary mapping tokens to integer indices.

        Args:
            dataset: The text to tokenize and build the vocabulary from.

        Returns:
            A dictionary mapping each token to a unique integer index.
        """
        if self._tokenization_mode == "character":
            if self._special_tokens is not None:
                special_tokens_regex = "|".join(
                    re.escape(token) for token in self._special_tokens
                )
                dataset = re.sub(special_tokens_regex, "", dataset)

            vocab = {
                token: index for index, token in enumerate(sorted(set(list(dataset))))
            }
        elif self._tokenization_mode == "word":
            vocab = {
                token: index
                for index, token in enumerate(sorted(set(list(dataset.split(" ")))))
            }
        else:
            raise ValueError(f"Unknown tokenization_mode: {self._tokenization_mode}")

        vocab["<unk>"] = len(vocab)
        if self._special_tokens is not None:
            for token in self._special_tokens:
                vocab[token] = len(vocab)

        return vocab

    def _compile_regex(self, special_tokens_regex: str) -> re.Pattern:
        """
        Compiles a regular expression used to distinguish special tokens
        from normal text during character-level tokenization.

        Args:
            special_tokens_regex: Regex pattern matching all special tokens.

        Returns:
            A compiled regular expression pattern with named groups
            corresponding to TokenType values.
        """
        token_specification = [
            (TokenType.SPECIAL, special_tokens_regex),
            (TokenType.NORMAL, r"."),
        ]
        regex = "|".join(f"(?P<{name}>{value})" for name, value in token_specification)

        return re.compile(regex, re.DOTALL)

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
            "special_tokens": self._special_tokens,
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
            tokenizer._special_tokens = data["special_tokens"]
            if tokenizer._special_tokens is not None:
                tokenizer._compiled_regex = tokenizer._compile_regex(
                    "|".join(re.escape(token) for token in tokenizer._special_tokens)
                )

            tokenizer._vocab_encode = data["vocab"]
            tokenizer._vocab_decode = {v: k for k, v in tokenizer._vocab_encode.items()}

            if logger:
                logger.info(f"Succesfully loaded tokenizer from {path}")

            return tokenizer

        except FileNotFoundError as e:
            if logger:
                logger.exception(f"File {path} not found")

            raise e
