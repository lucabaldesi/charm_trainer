#!/usr/bin/env python3


from autocommand import autocommand
import brain
import datetime
import numpy as np
import os
import read_IQ as riq
import signal
import torch
import torch.nn as nn
import torch.optim as optim


def print_stats(acc_mat):
    classes = acc_mat.shape[0]
    ones = np.ones((classes, 1)).squeeze(-1)

    corrects = np.diag(acc_mat)
    acc = corrects.sum()/acc_mat.sum()
    recall = (corrects/acc_mat.dot(ones)).round(4)
    precision = (corrects/ones.dot(acc_mat)).round(4)
    f1 = (2*recall*precision/(recall+precision)).round(4)

    print(f"Accuracy: {acc}")
    print(f"\t\tRecall\tPrecision\tF1")
    for c in range(classes):
        print(f"Class {c}\t\t{recall[c]}\t{precision[c]}\t\t{f1[c]}")


class EarlyExitException(Exception):
    def __str__(self):
        return "Received termination signal"


class CharmTrainer(object):
    def __init__(self, id_gpu="0", data_folder=".", batch_size=64, loaders=8):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))
        print(f"Training on {self.device}")
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.model = brain.CharmBrain().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_data = riq.IQDataset(data_folder=data_folder)
        self.train_data.normalize(torch.tensor([-3.1851e-06, -7.1862e-07]), torch.tensor([0.0002, 0.0002]))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=loaders, pin_memory=True)

        self.val_data = riq.IQDataset(data_folder=data_folder, validation=True)
        self.val_data.normalize(torch.tensor([-3.1851e-06, -7.1862e-07]), torch.tensor([0.0002, 0.0002]))
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=loaders, pin_memory=True)

        self.running = False


    def training_loop(self, n_epochs):
        for epoch in range(n_epochs):
            loss_train = 0.0
            for chunks, labels in self.train_loader:
                if not self.running:
                    raise EarlyExitException
                chunks = chunks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                output = self.model(chunks)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
            if epoch % 10 == 0:
                print(f"{datetime.datetime.now()} Epoch {epoch}, loss {loss_train/len(self.train_loader)}")
                self.validate(train=False)

    def validate(self, train=True):
        loaders = [('val', self.val_loader)]
        if train:
            loaders.append(('train', self.train_loader))

        for name, loader in loaders:
            correct = 0
            total = 0
            acc_mat = np.zeros((len(self.train_data.label), len(self.train_data.label)))

            with torch.no_grad():
                for chunks, labels in loader:
                    if not self.running:
                        raise EarlyExitException
                    chunks = chunks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    output = self.model(chunks)
                    _, predicted = torch.max(output, dim=1)
                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())
                    for i in range(labels.shape[0]):
                        acc_mat[labels[i]][predicted[i]] += 1

            print(f"{name} accuracy: {correct/total}")
            print_stats(acc_mat)

    def save_model(self, filename='charm.pt'):
        '''
        load your model with:
        >>> model = brain.CharmBrain()
        >>> model.load_state_dict(torch.load(filename))
        '''
        torch.save(self.model.state_dict(), filename)

    def execute(self, n_epochs):
        self.running = True
        try:
            self.training_loop(n_epochs)
            self.validate(train=True)
        except EarlyExitException:
            pass
        self.save_model()
        self.running = True
        print("[Done]")

    def exit_gracefully(self, signum, frame):
        self.running = False


@autocommand(__name__)
def charm_trainer(id_gpu="0", data_folder=".", n_epochs=100, batch_size=64, loaders=8):
    ct = CharmTrainer(id_gpu=id_gpu, data_folder=data_folder, batch_size=batch_size, loaders=loaders)
    ct.execute(n_epochs=100)
