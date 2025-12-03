import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

#############################################
# 1 — Configuration
#############################################

MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 768
HIDDEN_DIM = 128
NUM_CLASSES = 2
EPOCHS = 2
LR = 2e-5
BATCH_SIZE = 4

#############################################
# 2 — Dataset d'exemple (à remplacer plus tard)
#############################################

class SimpleDataset(Dataset):
    def __init__(self, tokenizer):
        texts = [
            "I love this movie!",
            "This product is terrible.",
            "Amazing experience.",
            "I hate this.",
        ]
        labels = [1, 0, 1, 0]

        self.tokens = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "label": self.labels[idx]
        }
        return item

#############################################
# 3 — LSTM Classifier
#############################################

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        out = self.fc(lstm_out[:, -1, :])
        return out

#############################################
# 4 — Entraînement
#############################################

def train():
    print("Loading tokenizer and BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    bert.eval()

    dataset = SimpleDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(EPOCHS):
        for batch in loader:
            with torch.no_grad():
                bert_out = bert(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).last_hidden_state

            logits = model(bert_out)
            loss = criterion(logits, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {loss.item():.4f}")

    print("Saving model to model.pth...")
    torch.save(model.state_dict(), "model.pth")
    print("Done.")

if __name__ == "__main__":
    train()
