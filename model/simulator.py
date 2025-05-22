import random

from datasets import load_dataset
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

from model.classifier import HumanAIClassifier
from model.helpers import compute_metrics


class BittensorRLLoopSimulator:
    def __init__(self, data: pd.DataFrame, classifier: HumanAIClassifier):
        self.limit = self.batch_size()
        self.batch_n = 0
        self.data = data
        self.classifier = classifier
        self.prev_f1 = None
        self.metrics = pd.DataFrame()

    def run(self):
        for i, row in self.data.iterrows():
            # prediction
            predict, confidence = self.classifier.classify_texts(row["text"])
            data.at[i, "predict"] = predict
            data.at[i, "confidence"] = confidence

            # little noise
            if random.randint(0, 100) < 7: # todo tweak
                data.at[i, "predict"] = 0 if predict == 1 else 1
            if random.randint(0, 100) < 4: # todo tweak
                data.at[i, "label"] = 0 if data.at[i, "label"] == 1 else 1

            # batching logic
            self.limit -= 1
            data.at[i, "batch"] = self.batch_n
            if self.limit == 0:
                self.batch_hook()
                self.batch_n += 1
                self.limit = self.batch_size()
        self.data.to_csv("rl.csv")
        # self.classifier.model.save_pretrained('../distilbert-rl-experiment')
        # self.classifier.tokenizer.save_pretrained('../distilbert-rl-experiment')
        self.metrics.to_csv("rl_metrics_per_batch.csv")

    def batch_hook(self):
        batch_data = self.data[self.data["batch"] == self.batch_n]
        metrics = compute_metrics(batch_data)
        metrics["batch_n"] = self.batch_n
        self.metrics = pd.concat([self.metrics, metrics.to_frame().T])
        f1 = metrics.get("f1", 0)
        if self.prev_f1 is None:
            self.prev_f1 = f1
            return
        # Compare with previous and train if incentive (f1) increased
        if f1 > self.prev_f1:
            print("🔁 F1 increased — training on this batch.")
            self.classifier.train(batch_data["text"].tolist() , batch_data["predict"].tolist(), batch_data["confidence"].tolist())
        else:
            print("⏭ No F1 improvement — skipping training.")
        self.prev_f1 = f1

    def batch_size(self, base: int = 400) -> int:
        while True:
            base += random.randint(-50, 50)
            if 0 != random.randint(0, 3):
                return base




if __name__ == "__main__":
    # model setup
    model_path = "../distilbert-ai-vs-human"
    model: DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # model
    dataset = load_dataset("ilyasoulk/ai-vs-human", split="train")
    df = dataset.to_pandas()
    human_df = pd.DataFrame({"text": df["human"], "label": 0}).dropna()
    ai_df = pd.DataFrame({"text": df["ai"], "label": 1}).dropna()

    data: pd.DataFrame = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # classifier

    classifier = HumanAIClassifier(model, tokenizer, device)


    simulation = BittensorRLLoopSimulator(data, classifier)

    simulation.run()
