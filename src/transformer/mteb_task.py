"""module for evaluating an mteb task"""
from typing import Callable, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from src.transformer.model import Transformer


def generate_combinations(pred_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates all combination of token similarity pairs, as well as the labels for each pair.
    Pairs contain two sentences, that are semantically similar and therefore are assigned the label 1.
    All other combinations """
    pred_pairs = pred_pairs.reshape(len(pred_pairs)//2, -1)
    no_similarity_indices = torch.combinations(torch.arange(len(pred_pairs.flatten())))
    labels = torch.tensor([1] * len(pred_pairs) + [-1] * len(no_similarity_indices))
    pred_combination = torch.vstack([pred_pairs,
                                     pred_pairs.flatten()[no_similarity_indices].reshape(len(no_similarity_indices), -1)
                                     ])
    return labels, pred_combination


class Trainer:
    def __init__(self, loss: Callable, epochs: int, model: Transformer, dataloader: DataLoader):
        self.loss = loss
        self.epochs = epochs
        self.model = model
        self.dataloader = dataloader
        self.optimizer = AdamW(model.parameters())

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for token in tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}", unit='batches'):
                token_pairs = token.to(self.device)
                pred_pairs = self.model(token_pairs)  # pairs to list
                self.loss(token_pairs)
                self.model.zero_grad()

    def loss_for_combinations(self, pairs):
        labels, pred_combination = generate_combinations(pairs)
        return self.loss(pred_combination, labels)


def main():
    all_nli_data = load_dataset("sentence-transformers/all-nli", "pair")

    model = Transformer()
    loss = nn.CosineEmbeddingLoss()  # loss suited for a dataset, where only positive simmilarities
    # are present. This means, the dataset contains pairs of similar sentences.

    pretrained_model_name = "Salesforce/xgen-7b-8k-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))  # initialize model embeddings with this size

    dataloader = DataLoader(
        all_nli_data,
        batch_size=64,
        shuffle=True)

    epochs = 5
    trainer = Trainer(loss, epochs, model, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
