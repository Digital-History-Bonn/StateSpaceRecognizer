import lightning
import torch
from torch import optim
from torch.nn import ConstantPad1d
from torch.nn.functional import cross_entropy, one_hot
from torchvision import transforms


def collate_fn(batch):
    """Custom collate function, that pads crops horizontally to fit them all in one tensor batch."""
    crops, targets, texts = zip(*batch)

    max_width = 0
    max_length = 0
    padded_crops = []
    padded_targets = []

    for crop, target in zip(crops, targets):
        width = crop.shape[-1]
        if width > max_width:
            max_width = width

        length = len(target)
        if length > max_length:
            max_length = length

    for crop, target in zip(crops, targets):
        if crop.shape[-1] < max_width:
            transform = transforms.Pad((max_width - crop.shape[-1], 0, 0, 0))
            padded_crops.append(transform(crop))
        if len(target) < max_length:
            # value 0 is always the '<PAD>' token
            transform = ConstantPad1d((0, max_length - len(target)), 0)
            padded_targets.append(transform(target))
    return torch.stack(padded_crops), torch.stack(padded_targets), texts


def one_hot_encoding(targets: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Handels one hot encoding of target to be usable for loss function.
    Args:
        targets: targets with shape[B,L]
        dim: Number of classes.
    Returns:
        torch.Tensor: targets with shape [B,C,L]
    """
    # pylint: disable-next=not-callable
    return torch.permute(one_hot(targets, num_classes=dim),
                         (0, 2, 1))


class SSMOCRTrainer(lightning.LightningModule):
    """Lightning module for image recognition training. Predict step returns a source object from the dataset as well as
    the softmax prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        image, target, _ = batch
        loss = self.run_model(image, target)
        self.log("train_loss", loss)
        return loss

    def run_model(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Predict input image and calculate loss. The target is modified, so that it consists out of start token for
        the same length as the encoder result. Only after the encoder results have been processed, the actual output
        starts.
        """
        start_token = self.model.tokenizer.single_token('<START>')

        pred = self.model(image, target)
        pred = torch.nn.functional.softmax(self.decoder(pred), dim=1)
        diff = len(pred) - len(target) # todo: batch dimension!
        target = torch.cat((torch.full([diff], start_token), target), 1)
        loss = cross_entropy(pred, one_hot_encoding(target, len(self.model.tokenizer)))
        return loss

    def validation_step(self, batch):
        image, target, _ = batch
        loss = self.run_model(image, target)
        self.log("val_loss", loss)

    def test_step(self, batch):
        image, target, _ = batch
        loss = self.run_model(image, target)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-05)
        return optimizer
