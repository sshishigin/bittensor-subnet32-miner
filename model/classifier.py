import torch


class HumanAIClassifier:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def predict_batch_safe(self, texts: list[str]):
        predicted_class_ids, confidence = self.classify_texts(texts)

        # Format for Subnet 32: [[label] * len(text.split()) for each text]
        formatted_preds = [
            [label] * len(text.split()) for label, text in zip(predicted_class_ids, texts)
        ]

        return formatted_preds, confidence

    def classify_texts(self, texts):
        # Tokenize batch
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Получаем предсказания и уверенность
        probs = torch.softmax(logits, dim=1)
        predicted_class_ids = torch.argmax(probs, dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()  # Уверенность по предсказанному классу

        return predicted_class_ids[0], confidences[0]

    def train(self, texts, predictions, confidences, lr=2e-5, batch_size=16, epochs=1):
        import torch
        from torch.nn import CrossEntropyLoss
        from torch.optim import AdamW

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(reduction="none")

        for epoch in range(epochs):
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_labels = torch.tensor(predictions[i:i + batch_size], dtype=torch.long).to(self.device)
                batch_weights = torch.tensor(confidences[i:i + batch_size], dtype=torch.float).to(self.device)

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits

                loss = loss_fn(logits, batch_labels)
                weighted_loss = (loss * batch_weights).mean()

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
