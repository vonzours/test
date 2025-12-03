import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

#############################################
# 1 — Config
#############################################

MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 768
HIDDEN_DIM = 128
NUM_CLASSES = 3     # TweetEval = positive, negative, neutral
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16
PATIENCE = 2   # early stopping patience

device = "cuda" if torch.cuda.is_available() else "cpu"

#############################################
# 2 — Dataset PyTorch Wrapper
#############################################

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels = torch.tensor(labels)
        self.tokens = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
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
        return self.fc(lstm_out[:, -1, :])

#############################################
# 4 — Train + Eval Loop
#############################################

def evaluate(model, bert, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    losses = []

    with torch.no_grad():
        for batch in loader:
            texts = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            bert_out = bert(input_ids=texts, attention_mask=masks).last_hidden_state
            logits = model(bert_out)

            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return sum(losses) / len(losses), acc, f1


def train():
    print("🔄 Loading TweetEval dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["label"]

    print("🔄 Loading tokenizer and BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    bert.to(device)
    bert.eval()

    # Freeze BERT layers
    for param in bert.parameters():
        param.requires_grad = False

    train_ds = TweetDataset(train_texts, train_labels, tokenizer)
    val_ds   = TweetDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0

    print("🚀 Training started!")
    for epoch in range(EPOCHS):

        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            texts = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                bert_out = bert(input_ids=texts, attention_mask=masks).last_hidden_state

            logits = model(bert_out)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)
        val_loss, acc, f1 = evaluate(model, bert, val_loader, criterion)

        print(f"Epoch {epoch+1} | Train loss = {avg_train_loss:.4f} | "
              f"Val loss = {val_loss:.4f} | Acc = {acc:.4f} | F1 = {f1:.4f}")

        # Early stopping + save best model
        if val_loss < best_loss:
            print("💾 Saving new best model...")
            best_loss = val_loss
            torch.save(model.state_dict(), "model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered.")
            break

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()
