import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        bonafide_scores = torch.softmax(
            logits.detach(),
            dim=-1,
        )[:, 1]
        self.scores.append(bonafide_scores.cpu())
        self.labels.append(labels.detach().cpu())

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        self(logits=logits, labels=labels)

    def reset(self):
        self.scores = []
        self.labels = []

    def compute(self):
        scores = torch.cat(self.scores).numpy()
        labels = torch.cat(self.labels).numpy()

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        eer, _ = compute_eer(
            bonafide_scores,
            spoof_scores,
        )
        return float(eer * 100)
