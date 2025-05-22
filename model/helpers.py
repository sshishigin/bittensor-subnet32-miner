import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model.simulator import tokenizer


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


def compute_metrics(batch):
    return pd.Series({
        'precision': precision_score(batch['label'], batch['predict'], zero_division=0),
        'recall': recall_score(batch['label'], batch['predict'], zero_division=0),
        'f1': f1_score(batch['label'], batch['predict'], zero_division=0),
        'accuracy': accuracy_score(batch['label'], batch['predict'])
    })
