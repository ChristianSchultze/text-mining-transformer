"""Test module for training components."""
import numpy as np
import torch
from src.transformer.model import Attention
from src.transformer.mteb_task import generate_combinations


class TestTransformer:
    """Test training components"""

    def test_combination(self):
        pairs = torch.tensor([1, 2, 4, 3, 5, 6])
        gt_labels = torch.tensor([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        gt_predictions = torch.tensor(
            [[1, 2], [4, 3], [5, 6], [1, 2], [1, 4], [1, 3], [1, 5], [1, 6], [2, 4], [2, 3], [2, 5], [2, 6], [4, 3],
             [4, 5], [4, 6], [3, 5], [3, 6], [5, 6]])
        labels, predictions = generate_combinations(pairs)

        assert np.all(labels == gt_labels)
        assert np.all(predictions == gt_predictions)
