import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

#############################################
# 1 — Mini dataset d'exemple
#############################################

class SimpleDataset(Dataset):
    def __init__(self, tokenizer):
        texts = [
            "I love this movie!",
            "This product is terrible.",
            "Amazing experience.",
            "I hate this."
        ]
        labels = [1, 0, 1, 0]

        self.labels = torch.tensor(labels)
        self.tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "label": self.labels[idx]
        }

#############################################
# 2 — LSTM Classifier
#############################################

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        return self.fc(lstm_out[:, -1, :])

#############################################
# 3 — Entraînement
#############################################

def train():
    print("Loading tokenizer and BERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased")
    bert.eval()

    # On gèle BERT
    for p in bert.parameters():
        p.requires_grad = False

    dataset = SimpleDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = LSTMClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    print("Training...")
    for epoch in range(3):
        losses = []
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

            losses.append(loss.item())

        print(f"Epoch {epoch+1} | Loss = {sum(losses)/len(losses):.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print("Done.")

if __name__ == "__main__":
    train()
