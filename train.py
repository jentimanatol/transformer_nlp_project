
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tokenizer import SimpleTokenizer
from model import TransformerClassifier
from data import load_data
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 256
batch_size = 16
epochs = 8
learning_rate = 3e-4

X_train, X_val, y_train, y_val, num_classes = load_data()

tokenizer = SimpleTokenizer(max_vocab_size=25000)
tokenizer.build_vocab(X_train)

X_train_ids = [tokenizer.encode(t, max_len) for t in X_train]
X_val_ids = [tokenizer.encode(t, max_len) for t in X_val]

train_dataset = TensorDataset(torch.tensor(X_train_ids), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val_ids), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = TransformerClassifier(
    vocab_size=len(tokenizer.word2idx),
    num_classes=num_classes,
    pad_id=tokenizer.word2idx["<PAD>"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    print(f"Validation Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
