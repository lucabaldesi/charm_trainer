#!/usr/bin/env python3

import math
import multiprocessing as mp
import random
import torch


def fetch(bundle):
     data, idx, queue = bundle
     queue.put(data[idx])


def fetch2(data, idx, queue):
     queue.put(data[idx])


class BetterDataLoader(object):
    def __init__(self, data, batch_size, shuffle, num_workers):
        self.data = data
        self.batch_n = batch_size
        self.shuffle = shuffle
        self.worker_n = num_workers
        self.workers = mp.Pool(self.worker_n)
        self.queue = mp.Queue()
        self.idxs = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.idxs)
        print(f"data len {len(self.data)}, batch size {self.batch_n}")
        print(f"lenght {len(self)}")

    def __len__(self):
        return int(math.ceil(len(self.data)/self.batch_n))

    def get_batch_idxs(self, i):
        if i<len(self):
            return self.idxs[i*self.batch_n:][:self.batch_n]
        raise IndexError

    def __getitem__(self, i):
        idxs = self.get_batch_idxs(i)

        #print(idxs)
        data = []
        for idx in idxs:
            data.append(self.data[idx])
 

        chunks = torch.stack([d for (d, l) in data])
        labels = torch.stack([l for (d, l) in data])

        return (chunks, labels)
