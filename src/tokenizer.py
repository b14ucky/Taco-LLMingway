from typing import Literal


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
            vocab = {token: index for index, token in enumerate(set(list(dataset)))}
        elif self._tokenization_mode == "word":
            vocab = {
                token: index
                for index, token in enumerate(set(list(dataset.split(" "))))
            }
        else:
            raise ValueError(f"Unknown tokenization_mode: {self._tokenization_mode}")

        vocab["<unk>"] = len(vocab)

        return vocab
