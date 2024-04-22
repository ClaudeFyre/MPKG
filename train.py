import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import *

def read_csv(dataset):
    # Load CSV files into pandas DataFrames
    KG1_df = pd.read_csv('data/'+ dataset +'KG1.csv')
    KG2_df = pd.read_csv('data/'+ dataset +'KG2.csv')
    entity_pair_df = pd.read_csv('data/'+ dataset +'entity_pair.csv')
    # Convert DataFrames to NumPy arrays
    KG1 = KG1_df.values
    KG2 = KG2_df.values
    entity_pair = entity_pair_df.values
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((KG1, KG2), axis=1), entity_pair, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

dataset = {"CN", "NELL", "PPI"}
X_train, X_test, y_train, y_test = read_csv(dataset[0])
# Create datasets
train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Initialize models
opt = {'num_feature': X_train.shape[1], 'num_class': num_classes, 'hidden_dim': 64,
       'input_dropout': 0.5, 'dropout': 0.5, 'cuda': torch.cuda.is_available()}
gnnq = GNNq(opt, adj)
gnnp = GNNp(opt, adj)
# Define and train your model
model = LogisticRegression()
model.fit(X_train, y_train)

# Move models to GPU if CUDA is available
if opt['cuda']:
    gnnq = gnnq.cuda()
    gnnp = gnnp.cuda()
    adj = adj.cuda()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_q = optim.Adam(gnnq.parameters(), lr=0.01)
optimizer_p = optim.Adam(gnnp.parameters(), lr=0.01)

for epoch in range(num_epochs):
    gnnq.train()
    gnnp.train()
    total_loss = 0

    for data, labels in train_loader:
        if opt['cuda']:
            data, labels = data.cuda(), labels.cuda()

        # Forward pass in GNNq
        optimizer_q.zero_grad()
        outputs_q = gnnq(data)
        loss_q = criterion(outputs_q, labels)
        loss_q.backward()
        optimizer_q.step()

        # Forward pass in GNNp using labels as features (assuming labels are also used as input features somehow)
        optimizer_p.zero_grad()
        outputs_p = gnnp(labels.float())  # Example usage; modify based on actual data structure
        loss_p = criterion(outputs_p, labels)
        loss_p.backward()
        optimizer_p.step()

        total_loss += loss_q.item() + loss_p.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    # Optionally, add validation and testing loops for model evaluation

def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            if opt['cuda']:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

evaluate(gnnq, test_loader)  # Evaluate GNNq
evaluate(gnnp, test_loader)  # Evaluate GNNp

# Saving
torch.save(gnnq.state_dict(), 'gnnq_model.pth')
torch.save(gnnp.state_dict(), 'gnnp_model.pth')

# Loading
gnnq.load_state_dict(torch.load('gnnq_model.pth'))
gnnp.load_state_dict(torch.load('gnnp_model.pth'))
