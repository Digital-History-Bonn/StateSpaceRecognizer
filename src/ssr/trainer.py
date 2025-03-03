"""Module for training related functions and the lightning module for the training setting."""
from typing import Tuple, Optional, List, Any

import Levenshtein
import lightning
import numpy as np
import torch
from torch import optim
from torch.nn import ConstantPad1d
from torch.nn.functional import cross_entropy
from torchvision import transforms

from ssr import Tokenizer

from ssr.mamba_recognizer import process_prediction


def collate_fn(batch: Tuple[List[torch.Tensor], ...]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate function, that pads crops horizontally to fit them all in one tensor batch."""
    crops, targets, texts = zip(*batch)

    max_length, max_width = get_max_size(crops, targets) # type: ignore
    padded_crops, padded_targets = pad_data(crops, max_length, max_width, targets) # type: ignore

    return torch.stack(padded_crops), torch.stack(padded_targets), texts # type: ignore


def pad_data(crops: List[torch.Tensor], max_length: int, max_width: int, targets: List[torch.Tensor]) -> tuple[list[Any], list[Any]]:
    """
    Pad crops and targets respectively to match the largest width/length and enable stacking them to an input and
    target tensor with all data of this batch.
    """
    padded_crops = []
    padded_targets = []
    for crop, target in zip(crops, targets):
        if crop.shape[-1] < max_width:
            transform = transforms.Pad((0, 0, max_width - crop.shape[-1], 0))
            padded_crops.append(transform(crop))
        else:
            padded_crops.append(crop)
        if len(target) < max_length:
            # value 0 is always the '<PAD>' token
            transform = ConstantPad1d((0, max_length - len(target)), 0)
            padded_targets.append(transform(target))
        else:
            padded_targets.append(target)
    return padded_crops, padded_targets


def get_max_size(crops: List[torch.Tensor], targets: List[torch.Tensor]) -> Tuple[int, int]:
    """
    Determine maximum crop width and target length.
    Args:
        targets (List[torch.Tensor]): List of tensors shape [L]
        crops (List[torch.Tensor]): List of tensors shape [1,H,W]
    """
    max_width = 0
    max_length = 0
    for crop, target in zip(crops, targets):
        width = crop.shape[-1]
        if width > max_width:
            max_width = width

        length = len(target)
        if length > max_length:
            max_length = length
    return max_length, max_width


def calculate_ratio(data_list: List[Tuple[int, int]]) -> float:
    """Calculate ratio from list containing values and lengths."""
    data_ndarray = np.array(data_list)
    sums = np.sum(data_ndarray, axis=0)
    ratio: float = sums[0] / sums[1]
    return ratio


class SSMOCRTrainer(lightning.LightningModule):
    """Lightning module for image recognition training. Predict step returns a source object from the dataset as well as
    the softmax prediction."""

    def __init__(self, model, batch_size: int, tokenizer: Tokenizer):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def training_step(self, batch, _):
        self.model.train()
        image, target, _ = batch
        loss, _ = self.run_model(image, target)
        self.log("train_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=True, on_epoch=True,
                 on_step=True)
        return loss

    def run_model(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict input image and calculate loss. The target is modified, so that it consists out of start token for
        the same length as the encoder result. Only after the encoder results have been processed, the actual output
        starts.
        """
        self.model.device = self.device
        image = image.cuda(self.device)
        target = target.cuda(self.device)
        pad_token = self.tokenizer.single_token('<PAD>')

        pred = self.model(image, target)
        diff = pred.shape[-1] - target.shape[-1]
        target = torch.cat((torch.full((target.shape[0], diff - 1), pad_token).cuda(self.device), target), 1)
        target = torch.cat((target, torch.full((target.shape[0], 1), pad_token).cuda(self.device)), 1)
        loss = cross_entropy(pred, target, ignore_index=pad_token)
        return loss, pred[:, :, diff - 1:]

    def validation_step(self, batch: torch.Tensor, _):
        """Evaluate validation dataset"""
        self.model.eval()
        self.evaluate_prediction(batch, "val")

    def test_step(self, batch: torch.Tensor, _):
        """Evaluate test dataset"""
        self.model.eval()
        self.evaluate_prediction(batch, "test")

    def evaluate_prediction(self, batch: torch.Tensor, name: str):
        """Evaluate input batch and log with supplied name tag.
        Predicts model, converts output tokens to text and calculates levenshtein distance."""
        image, target, texts = batch
        loss, pred = self.run_model(image, target)
        self.log(f"{name}_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=False)
        pred = process_prediction(self.tokenizer.single_token('<NAN>'), pred, self.model.confidence_threshold)
        self.levenshtein_distance(pred, texts, name)

    def levenshtein_distance(self, pred: torch.Tensor, targets: List[str], name: str) -> None:
        """Calculate Levenshtein distance between prediction and targets for each line. Distance is then averaged per
        character."""
        distance_list = []
        for target, i in enumerate(targets):
            pred_line = self.tokenizer.to_text(pred[i])
            gt_line = targets
            # print(f"pred: {pred_line}\n gt: {gt_line} \n \n")
            distance = Levenshtein.distance(gt_line, pred_line)
            distance_list.append((distance, (len(gt_line) + len(pred_line))))
        ratio = calculate_ratio(distance_list)
        self.log(f"{name}_levenshtein", ratio, batch_size=self.batch_size, prog_bar=True, on_epoch=True,
                 on_step=True)

    def configure_optimizers(self):
        """Configure AdamW optimizer from config."""
        optimizer = optim.AdamW(self.parameters(), lr=1e-04, weight_decay=1e-05) #todo: add config
        return optimizer
