"""Test module for the mamba ssm recognizer."""
import os
from abc import ABC
from pathlib import Path

import pytest
import torch
import yaml

from src.ssr.mamba_recognizer import Recognizer, image_to_sequence, FeedForward
from src.ssr.tokenizer import Tokenizer


@pytest.mark.datafiles("resources/recognizer.yml")
class TestRecognizer:
    """Tests for the mamba ssm recognizer."""

    @pytest.fixture(scope='class', autouse=True)
    def setup(self):
        # Initialize shared attributes once for the class
        pytest.ressource_path = Path(os.path.join(os.path.dirname(__file__), "tests/resources"))
        with open(pytest.ressource_path / "recognizer.yml", "r") as file:
            pytest.cfg = yaml.safe_load(file)
        pytest.tokenizer = TestTokenizer()
        pytest.recognizer = Recognizer(pytest.cfg, pytest.tokenizer)

    def test_image_to_sequence(self):
        """Image to sequence conversion is done by flattening the C and H dimension of an [B,C,H,W] image.
        The resulting [B,C,L] sequence is required to preserve the order in the L dimension."""
        data = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
        groung_truth = torch.tensor([[[1, 2],
                                      [3, 4],
                                      [5, 6],
                                      [7, 8]],
                                     [[9, 10],
                                      [11, 12],
                                      [13, 14],
                                      [15, 16]]])
        assert image_to_sequence(data) == groung_truth

    def test_feed_forward(self):
        """Test feed forward, forward pass."""
        model_dim = 512
        ground_truth = (1, model_dim, 17)

        data = torch.zeros(ground_truth)
        ff = FeedForward(model_dim=model_dim)
        result = ff(data)
        assert result.shape == ground_truth

    def test_layer(self):
        """Test the SSMLayer forward pass. One test for the encoder version with downscaling and one for the decoder
        version without downscaling."""
        shape = (2, 160, 50)
        input_data = torch.ones(shape)

        result = pytest.recognizer.encoder.layers[0](input_data)

        assert result.shape == (2, 320, 25)

        shape = (2, 1280, 6)
        input_data = torch.ones(shape)

        result = pytest.recognizer.decoder.layers[0](input_data)

        assert result.shape == (2, 1280, 6)

    def test_encoder(self):
        """Test the encoder forward pass."""
        shape = (2, 1, 40, 200)
        input_data = torch.ones(shape)

        result = pytest.recognizer.encoder(input_data)

        assert result.shape == (2, 1280, 6)

    def test_recognizer(self):
        """Test the recognizer forward pass."""
        shape = (2, 1, 40, 200)
        input_data = torch.ones(shape)
        target = torch.tensor([5, 5, 4, 4, 6, 6])

        result = torch.nn.functional.softmax(pytest.recognizer(input_data, target), dim=1)
        result_batch = [pytest.tokenizer.to_text(result[0]), pytest.tokenizer.to_text(result[1])]

        assert result.shape == (2, 12)
        assert len(result_batch[0]) == 12 and len(result_batch[1]) == 12


class TestTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(['<PAD>', '<START>', '<NAN>', '<END>', 'a', 'b', 'c'])

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
