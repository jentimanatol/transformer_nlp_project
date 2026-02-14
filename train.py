
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tokenizer import SimpleTokenizer
from model import TransformerClassifier
from data import load_data
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
max_len = 256
batch_size = 16
epochs = 8
learning_rate = 3e-4

# Load data
X_train, X_val, y_train, y_val, num_classes = load_data()

# Tokenizer
tokenizer = SimpleTokenizer(max_vocab_size=25000)
tokenizer.build_vocab(X_train)

# Encode
X_train_ids = [tokenizer.encode(text, max_len) for text in X_train]
X_val_ids = [tokenizer.encode(text, max_len) for text in X_val]

X_train_tensor = torch.tensor(X_train_ids)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val_ids)
y_val_tensor = torch.tensor(y_val)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = TransformerClassifier(vocab_size=len(tokenizer.word2idx), num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Validation Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
