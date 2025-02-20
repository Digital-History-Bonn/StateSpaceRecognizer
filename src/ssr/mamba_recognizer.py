"""Module for mamba based OCR model."""
from typing import List, Optional
from unittest.mock import inplace

import torch
from torch import nn
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams
from torchvision.transforms.functional import normalize

from ssr.tokenizer import Tokenizer


def create_empty_dict(length: int):
    return {None for _ in range(length)}


def process_prediction(nan_token: int, pred: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Applies softmax and extracts token with the highest probability, provided it is above threshold."""
    result_batch = torch.nn.functional.softmax(pred, dim=1)
    max_tensor, argmax = torch.max(result_batch, dim=1)
    argmax = argmax.type(torch.uint8)
    argmax[max_tensor < threshold] = nan_token
    return argmax.detach().cpu() # type: ignore


class Recognizer(nn.Module):
    """Implements OCR model composed of a visual encoder and a sequence decoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.vocab_size = cfg["vocabulary"]["size"]
        self.cfg = cfg
        self.encoder = Encoder(cfg["encoder"])
        self.embedding = nn.Embedding(self.vocab_size, cfg["encoder"]["block"]["dim"] * self.encoder.expansion_factor,
                                      padding_idx=0)  #todo: config better
        self.decoder = Decoder(cfg["decoder"], self.encoder.expansion_factor, self.vocab_size)

        self.confidence_threshold = cfg["confidence_threshold"]

        # initialize normalization
        self.register_buffer("means", torch.tensor([0.443]))  # gray scale normalization for image data.
        self.register_buffer("stds", torch.tensor([0.226]))  # todo: put this into model config
        self.normalize = normalize

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with the target sequence as additional input for the decoder, after
        processed through the embedding layer.
        Args:
            image: Image data with shape[B,C,H,W]
            target: token ids with shape [B,L]
        Returns:
            torch.Tensor: with shape [B,C,L]
            """
        image = self.normalize(image, self.means, self.stds)
        encoder_tokens = self.encoder(image)
        target_embeddings = torch.permute(self.embedding(target), (0, 2, 1))
        decoder_tokens = self.decoder(torch.cat((encoder_tokens, target_embeddings), 2))
        return decoder_tokens  # type:ignore

    def inference(self, image: torch.Tensor, batch_size: int, tokenizer: Tokenizer) -> List[str]:
        """Forward pass for inference. After image encoding the output sequence is generated.
        Args:
            image(torch.Tensor): Image data with shape[B,C,H,W]
            """
        image = self.normalize(image, self.means, self.stds)
        encoder_tokens = self.encoder(image)
        return self.generate(encoder_tokens, batch_size, tokenizer)

    def generate(self, encoder_tokens: torch.Tensor, batch_size: int, tokenizer: Tokenizer) -> List[str]:
        """Generate OCR output at inference time. This is done
        in an autoregressive way, passing each output back to the model, and ends with an end (or padding) token.
        Args:
            encoder_tokens(torch.Tensor): encoder processed tokens with shape [B,C,L]
        """
        # TODO: positional encodings?
        start_token = tokenizer.single_token('<START>')
        end_token = tokenizer.single_token('<END>')
        nan_token = tokenizer.single_token('<NAN>')
        result_tokens = [[start_token]] * batch_size
        start_token = self.embedding(start_token)
        start_list = []
        for i in range(batch_size):
            start_list.append(start_token.clone())
        input_batch = torch.stack(start_list)

        self.decoder.allocate_inference_cache(batch_size, 0)
        self.decoder(encoder_tokens)
        while True:
            pred = self.decoder(input_batch)

            result_tensor = process_prediction(nan_token, pred, self.confidence_threshold)

            for i, result in enumerate(result_tensor.tolist()):
                result_tokens[i] += [result]
            input_batch = self.embedding(result_tensor)

            if all(result[-1] == end_token for result in result_tokens):
                break
        return [tokenizer.to_text(torch.tensor(result)) for result in result_tokens]

    def load(self, path: Optional[str], device: str) -> None:
        """
        load the model weights
        """
        if path is None:
            return
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()


def image_to_sequence(image: torch.Tensor) -> torch.Tensor:
    """
    Merge channel and height dimension of an image, such that the resulting sequence preserves the x-axis order.
    Args:
        image(torch.Tensor): [B,C,H,W]

    Returns:
        torch.Tensor: sequence of [B,C,L]

    """
    return image.flatten(1, 2)


class Encoder(nn.Module):
    """Implements encoder with multiple layers of mamba blocks and an initial downscaling convolution."""

    def __init__(self, cfg: dict):
        """Creates ssm layers with an initial downscaling 2d convolution."""
        super().__init__()
        layers: List[int] = cfg["layers"]["num_blocks"]

        channel_1 = cfg["channels"][0]
        self.conv1 = nn.Conv2d(
            1,
            channel_1,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(channel_1)

        channel_2 = cfg["channels"][1]
        self.conv2 = nn.Conv2d(
            channel_1,
            channel_2,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(channel_2)

        self.downsample = nn.Sequential(nn.Conv2d(
            1,
            channel_2,
            kernel_size=1,
            stride=4,
            bias=False,
        ), torch.nn.BatchNorm2d(channel_2))

        self.relu = torch.nn.ReLU(inplace=True)

        expansion_factor = int(cfg["channels"][1] // 4)  # todo: config better
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(SSMLayer(layer, expansion_factor, cfg["layers"]["downscale"], cfg["block"]))
            if cfg["layers"]["downscale"]:
                expansion_factor *= 2
        self.expansion_factor = expansion_factor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Executes encoder layers
        Args:
            image(torch.Tensor): Image with shape [B,C,H,W]
        Returns:
            torch.Tensor: tokens with shape [B,C,L]
        """
        residual = image.clone()
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.conv2(image)
        image = self.bn2(image)
        residual = self.downsample(residual)
        image = self.relu(image + residual)

        tokens = image_to_sequence(image)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens  # type:ignore


class Decoder(nn.Module):
    """Implements decoder with multiple layers of mamba blocks, a language head and embeddings autoregressive
    processing of previous outputs."""

    def __init__(self, cfg: dict, encoder_expansion: int, vocab_size: int):
        """Creates ssm layers with an initial downscaling 2d convolution."""
        super().__init__()
        layers: List[int] = cfg["layers"]["num_blocks"]

        self.layers = nn.ModuleList()
        expansion_factor = encoder_expansion
        for layer in layers:
            self.layers.append(SSMLayer(layer, expansion_factor, cfg["layers"]["downscale"], cfg["block"]))
            if cfg["layers"]["downscale"]:
                expansion_factor *= 2
        self.bn = torch.nn.BatchNorm1d(vocab_size)
        self.lm_head = torch.nn.Conv1d(cfg["block"]["dim"] * expansion_factor, vocab_size, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes decoder layers
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            torch.Tensor: tokens with shape [B,C,L]
        """

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.lm_head(tokens)
        tokens = self.bn(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int) -> None:
        """
        Returns the inference parameters for all layers. This allows for efficient inference.
        """
        param_example = next(iter(self.parameters()))
        dtype = param_example.dtype
        for layer in self.layers:
            layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)


class SSMLayer(nn.Module):
    """Implements a Layer consisting of multiple mamba blocks and an initial downscaling convolution ."""

    def __init__(self, num_blocks: int, layer_factor: int, downscale: bool, block_config: dict):
        """Creates multiple mamba blocks and an initial downscaling convolution."""
        super().__init__()
        channels = block_config["dim"] * layer_factor
        self.downscale = downscale
        if downscale:
            self.conv = nn.Conv1d(
                channels,
                channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.downsample = nn.Sequential(nn.Conv1d(
                channels,
                channels * 2,
                kernel_size=1,
                stride=2,
                bias=False,
            ), torch.nn.BatchNorm1d(channels * 2))
            channels *= 2
        self.norm = torch.nn.BatchNorm1d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SSMBlock(block_config, channels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes SSM Layer consisting out of multiple mamba blocks and an initial downscaling convolution.
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            tokens: tokens with shape [B,C,L]
        """

        if self.downscale:
            residual = tokens.clone()
            tokens = self.conv(tokens)
            tokens = self.norm(tokens)
            residual = self.downsample(residual)
            tokens = self.relu(tokens + residual)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype) -> None:
        """
        Returns the inference parameters for all blocks. This allows for efficient inference.
        """
        for block in self.blocks:
            block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)


class SSMBlock(nn.Module):
    """Implements a Layer consisting of multiple mamba blocks and an initial downscaling convolution."""

    def __init__(self, cfg: dict, channels: int):
        """Creates a mamba block wrapped with batch norm and a residual connection, followed by a fully
        connected layer."""
        super().__init__()

        self.has_feed_forward = cfg["feed_forward"]

        self.ssm = (Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            # todo: tie channels to expansion and image height
            d_model=channels,  # Model dimension d_model
            d_state=cfg["state"],  # SSM state expansion factor, typically 64 or 128
            d_conv=cfg["conv_width"],  # Local convolution width
            expand=cfg["expand"],  # Block expansion factor
            headdim=cfg["dim"],  # d_model needs to be a multiple of headdim
            layer_idx=0  # default id for accessing inference cache.
        ))
        self.norm = torch.nn.BatchNorm1d(channels)

        if self.has_feed_forward:
            self.feed_forward = FeedForward(channels, channels * 2)
        self.inference_params: Optional[InferenceParams] = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes SSM Block consisting out of multiple mamba blocks and an initial downscaling convolution.
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            tokens: tokens with shape [B,C,L]
        """

        residual = tokens.clone()

        tokens = torch.permute(tokens, (0, 2, 1))  # mamba block needs shape of [B,L,C]
        tokens = self.ssm(tokens, inference_params=self.inference_params)
        tokens = torch.permute(tokens, (0, 2, 1))
        tokens = self.norm(tokens) + residual

        if self.has_feed_forward:
            tokens = self.feed_forward(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype) -> None:
        """
        Returns the ssm and conv state of the mamba block. This allows for efficient inference.
        """
        inference_cache = {0: self.ssm.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)}
        self.inference_params = InferenceParams(
            max_seqlen=max_seqlen,  # this is obsolete in this implementation, but necessary for compatibility.
            max_batch_size=batch_size,
            key_value_memory_dict=inference_cache,
        )


class FeedForward(torch.nn.Module):
    """Implements feed-forward layer with one hidden layer and a residual connection with add + norm."""

    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_in = torch.nn.Conv1d(model_dim, hidden_dim, 1)
        self.linear_out = torch.nn.Conv1d(hidden_dim, model_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(model_dim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes feed forward layer with residual connection and norm.
        """
        residual = tokens.clone()
        result = self.linear_in(tokens)
        result = self.bn1(result)
        result = self.relu(result)
        result = self.linear_out(result)
        result = self.bn2(result)
        return result + residual  # type:ignore
