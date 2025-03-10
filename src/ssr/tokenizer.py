"""Module for abstract class Tokenizer"""
from abc import ABC, abstractmethod
from typing import Dict

import torch


class Tokenizer(ABC):
    """Abstract tokenizer class, enforcing pad, start, nan and end tokens."""
    def __init__(self, alphabet: Dict[str, int]):
        """
        Tokenizer for OCR.
        Args:
            alphabet(Dict[str]): alphabet for tokenization. '<PAD>', '<START>', '<NAN>', '<END>' token are
            required to have indices 0,1,2,3."""
        assert alphabet['<PAD>'] == 0 and alphabet['<START>'] == 1 and alphabet['<NAN>'] == 2 and alphabet[
            '<END>'] == 3, ("Tokenizer alphabet is required to have '<PAD>', '<START>', "
                            "'<NAN>', '<END>' tokens with indices 0,1,2,3.")
        self.alphabet = alphabet

    @abstractmethod
    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenizes a sequence.
        Args:
            text(str): text to be tokenized.

        Returns:
            torch.Tensor: 1d tensor with token ids.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of alphabet."""

    @abstractmethod
    def single_token(self, input_str: str) -> int:
        """
        Tokenizes a single character. This can include returning the index of a start, end or nan token.
        Args:
            input_str(str): text to be tokenized.

        Returns:
            int: token id.
        """

    @abstractmethod
    def single_token_to_text(self, token_id: int) -> str:
        """
         Converts single token id back to text.
         Args:
            token_id(int): token id.

        Returns:
            str: string representation of token.
         """

    @abstractmethod
    def to_text(self, token_ids: torch.Tensor) -> str:
        """
        Converts tensor with token ids back to text.
        Args:
            token_ids(torch.Tensor): torch tensor with token ids.

        Returns:
            str: Text representation of tokens.
        """
