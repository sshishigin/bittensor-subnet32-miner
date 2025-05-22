import time

import bittensor as bt
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


import detection
from model.classifier import HumanAIClassifier

from detection.base.miner import BaseMinerNeuron



class Miner(BaseMinerNeuron):
    def __init__(self, config=None, model=None):
        super(Miner, self).__init__(config=config)
        self.model = model
        self.load_state()
        self.predictions_log = pd.DataFrame()
        self.batch_n = 0
        self.last_incentive = 0


    async def forward(
        self, synapse: detection.protocol.TextSynapse
    ) -> detection.protocol.TextSynapse:

        start_time = time.time()

        input_data = synapse.texts
        bt.logging.info(f"Amount of texts recieved: {len(input_data)}")

        try:
            preds, confidence = self.model.predict_batch_safe(input_data)
        except Exception as e:
            bt.logging.error(e)
            preds = [0] * len(input_data)
            confidence = 0

        preds = [[pred] * len(text.split()) for pred, text in zip(preds, input_data)]
        bt.logging.info(f"Made predictions in {int(time.time() - start_time)}s")
        self.predictions_log = pd.concat(
            [
                self.predictions_log,
                pd.DataFrame(
                    {
                        "text": input_data,
                        "predict": preds,
                        "confidence": confidence,
                        "batch":self.batch_n,
                        "incentive": self.last_incentive
                    }
                )
            ]
        )
        synapse.predictions = preds
        incentive = self.get_incentive()
        if self.last_incentive != incentive:
            bt.logging.info(f"batch_n={self.batch_n}, incentive={incentive}")
            self.last_incentive = incentive
            self.predictions_log.to_csv(f"miner_logs_{self.batch_n}.csv")
            self.predictions_log = pd.DataFrame()
            self.batch_n += 1
        return synapse

    def get_incentive(self) -> float:
        return self.subtensor.metagraph(32).emission[self.uid].item()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.predictions_log.to_csv("miner_logs_last.csv")
        super(Miner, self).__exit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    model_path = "../distilbert-ai-vs-human"
    model: DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = HumanAIClassifier(model, tokenizer, device)

    with Miner(classifier) as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(30)
