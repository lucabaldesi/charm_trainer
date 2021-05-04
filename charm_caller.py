#!/usr/bin/env python3

from autocommand import autocommand
from torch.utils.tensorboard import SummaryWriter
import brain
import datetime
import numpy as np
import os
import read_IQ as riq
import signal
import torch
import charm_trainer as ct
import better_data_loader as bdl


def simple_validator(device, model_file, data_folder, chunk_size):
    val_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, validation=True)
    val_data.normalize(torch.tensor([-3.1851e-06, -7.1862e-07]), torch.tensor([0.0002, 0.0002]))

    model = brain.CharmBrain(chunk_size)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    tot = 0
    correct = 0
    with torch.no_grad():
        for chunk, label in val_data:
            chunk = chunk.to(device, non_blocking=True)
            output = model(chunk.unsqueeze(0))
            _, predicted = torch.max(output, dim=1)
            print(output)
            print(predicted)
            print(label)
            if predicted.item() == label:
                print("yeah")
                correct += 1
            tot += 1
    print(f"Accuracy: {correct/tot}")


def validator(device, model_file, data_folder, chunk_size):
    val_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, validation=True)
    val_data.normalize(torch.tensor([-3.1851e-06, -7.1862e-07]), torch.tensor([0.0002, 0.0002]))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = bdl.BetterDataLoader(val_data, batch_size=32, shuffle=True, num_workers=3)

    model = brain.CharmBrain(chunk_size)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    tot = 0
    correct = 0
    with torch.no_grad():
        for chunks, labels in val_loader:
            chunks = chunks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(chunks)
            _, predicted = torch.max(output, dim=1)
            correct += int((predicted == labels).sum())
            tot += labels.shape[0]
        print(f"Accuracy: {correct/tot}")


def trainer_validator(id_gpu, data_folder, model_file, chunk_size=24576):
    ct = CharmTrainer(id_gpu=id_gpu, data_folder=data_folder, batch_size=512, chunk_size=chunk_size, sample_stride=0,
                      loaders=20)
    ct.running = True
    ct.model.load_state_dict(torch.load(model_file))
    ct.model.to(ct.device)
    ct.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=loaders, pin_memory=True)
    ct.validate(0)


@autocommand(__name__)
def charm_caller(id_gpu, model_file, data_folder, chunk_size=24576):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
    device = (torch.device('cuda') if torch.cuda.is_available()
             else torch.device('cpu'))

    simple_validator(device, model_file, data_folder, chunk_size)  ## OK! consistent result: 0.33600939351738074
    #validator(device, model_file, data_folder, chunk_size)
    #trainer_validator(id_gpu, data_folder, model_file, chunk_size)
