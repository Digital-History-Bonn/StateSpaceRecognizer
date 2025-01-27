"""Test module for the mamba ssm recognizer."""
import os
from abc import ABC

import pytest
import torch
import yaml

from src.ssr.mamba_recognizer import Recognizer
from src.ssr.tokenizer import Tokenizer


@pytest.mark.datafiles("resources/recognizer.yml")
class TestRecognizer:
    """Tests for the mamba ssm recognizer."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, cls):
        # Initialize shared attributes once for the class
        ressource_path = os.path.join(os.path.dirname(__file__), "tests/resources")
        with open(ressource_path, "r") as file:
            cls.cfg = yaml.safe_load(file)
        cls.tokenizer = TestTokenizer()
        cls.recognizer = Recognizer(cls.cfg, cls.tokenizer)

    def test_recognizer(self):
        """Test the recognizer forward pass."""
        shape = (2, 1, 100, 200)
        input_data = torch.ones(shape)
        target = torch.tensor([5, 5, 4, 4, 6, 6])

        result = torch.nn.functional.softmax(self.recognizer(input_data, target), dim=1)
        result_batch = [self.tokenizer.to_text(result[0]), self.tokenizer.to_text(result[1])]

        assert result.shape == (2,12)
        assert len(result_batch[0]) == 12 and len(result_batch[1]) == 12


class TestTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(['<PAD>', '<START>', '<NAN>', '<END>','a','b','c'])

    def __len__(self) -> int:
        """
        Returns:
            int: the number of tokens
        """
        return len(self.alphabet)

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenizes a sequence.
        Args:
            text(str): text to be tokenized.

        Returns:
            torch.Tensor: 1d tensor with token ids.
        """
        result = []
        for char in text:
            result.append(self.single_token(char))
        return torch.tensor(result)

    def single_token(self, input: str) -> int:
        """
        Tokenizes a single character. This can include returning the index of a start, end or nan token.
        Args:
            input(str): text to be tokenized.

        Returns:
            int: token id.
        """
        return self.alphabet.index(input)

    def single_token_to_text(self, token_id: int) -> str:
        """
         Converts single token id back to text.
         Args:
            token_id(int): token id.

        Returns:
            str: string representation of token.
         """
        return self.alphabet[token_id]

    def to_text(self, token_ids: torch.Tensor) -> str:
        """
        Converts tensor with token ids back to text.
        Args:
            token_ids(torch.Tensor): torch tensor with token ids.

        Returns:
            str: Text representation of tokens.
        """
        result = ""
        for token_id in token_ids.tolist():
            result += self.single_token_to_text(token_id)
        return result
