#!/usr/bin/env python3


import brain
import datetime
import read_IQ as riq
import torch
import torch.nn as nn
import torch.optim as optim


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on {device}")


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(n_epochs):
        loss_train = 0.0
        for chunks, labels in train_loader:
            chunks = chunks.to(device)
            labels = labels.to(device)
            output = model(chunks)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch % 10 == 0:
            print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(train_loader)}")


def validate(model, train_loader, val_loader):
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for chunks, labels in loader:
                chunks = chunks.to(device)
                labels = labels.to(device)
                output = model(chunks)
                _, predicted = torch.max(output, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print(f"{name} accuracy: {correct/total}")


train_data = riq.IQDataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_data = riq.IQDataset(validation=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)

model = brain.CharmBrain().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(n_epochs = 100,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              train_loader = train_loader)

validate(model, train_loader, val_loader)
