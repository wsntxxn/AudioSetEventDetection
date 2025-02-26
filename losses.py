import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipBce(nn.Module):

    def forward(self, output):
        return F.binary_cross_entropy(
            output["clipwise_output"].clamp(1e-7, 1.), output["weak_target"])


class FrameBce(nn.Module):

    def forward(self, output):
        length = min(output["framewise_output"].shape[1],
                     output["strong_target"].shape[1])
        output["framewise_output"] = output["framewise_output"][:, :length, :]
        output["strong_target"] = output["strong_target"][:, :length, :]
        return F.binary_cross_entropy(
            output["framewise_output"].clamp(1e-7, 1.),
            output["strong_target"])


class ClipFrameBceLoss(nn.Module):
    def __init__(self, alpha=0.5, auto=False):
        super().__init__()
        self.clip_fn = ClipBce()
        self.frame_fn = FrameBce()
        self.alpha = alpha
        self.auto = auto

    def forward(self, output):
        clip_loss = self.clip_fn(output)
        frame_loss = self.frame_fn(output)
        if self.auto and self.training:
            return clip_loss / clip_loss.detach() + \
                frame_loss / frame_loss.detach()
        else:
            return self.alpha * clip_loss + (1 - self.alpha) * frame_loss
