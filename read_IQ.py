#!/usr/bin/env python3

import io
import numpy as np
import torch


'''
sample: a couple Inphase Quadrature (float, float)
chunk_size: the sample number per chunk
chunk: the numpy array we deal with
chunk_num: total number of chunks to serve
chunk_offset: the number of initial chunks to skip (for creating sub datasets)

dtype '<f4' means 'little endian floating variable of 4 bytes'
'''
class IQData(object):
    def __init__(self, filename, label, chunk_size, chunk_num, chunk_offset=0):
        self.filename = filename
        self.file = open(self.filename, 'rb')
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.label = label
        self.chunk_offset = chunk_offset
        self.normalization = None

    def __len__(self):
        return self.chunk_num

    def seek_record(self, idx):
        self.file.seek((idx+self.chunk_offset)*self.chunk_size*2*4)

    def __getitem__(self, idx):
        self.seek_record(idx)
        d = np.fromfile(self.file, dtype='<f4', count=self.chunk_size*2)
        d = np.transpose(d.reshape((self.chunk_size, 2)))
        d = torch.from_numpy(d)
        if self.normalization:
            d = (d - self.normalization[0]) / self.normalization[1]
        return (d, self.label)

    def normalize(self, mean, std):
        self.normalization = (mean.unsqueeze(-1), std.unsqueeze(-1))


class IQDataset(object):
    def __init__(self, chunk_size=20000, validation=False):
        self.label = {0: 'clear', 1: 'LTE', 2: 'WiFi'}
        self.dataset = []
        if not validation:
            chunks_per_dataset = 1200000000//chunk_size
            #chunks_per_dataset = 1000
            self.dataset.append(IQData("clear.bin", label=0,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData("LTE_HT_DL.bin", label=1,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("LTE_HT_UL.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("LTE_LT.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("LTE_ZT.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData("WIFI_HT_DL.bin", label=2,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("WIFI_HT_UL.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("WIFI_LT.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            #self.dataset.append(IQData("WIFI_ZT.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset))
        else:
            offset = 1200000000//chunk_size
            chunks_per_dataset = 600000000//chunk_size
            #chunks_per_dataset = 1000
            self.dataset.append(IQData("clear.bin", label=0,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData("LTE_HT_DL.bin", label=1,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            #self.dataset.append(IQData("LTE_HT_UL.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
            #self.dataset.append(IQData("LTE_LT.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
            #self.dataset.append(IQData("LTE_ZT.bin", label=1,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
            self.dataset.append(IQData("WIFI_HT_DL.bin", label=2,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            #self.dataset.append(IQData("WIFI_HT_UL.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
            #self.dataset.append(IQData("WIFI_LT.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
            #self.dataset.append(IQData("WIFI_ZT.bin", label=2,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset,
            #                           chunk_offset=offset))
        self.chunks_per_dataset = chunks_per_dataset
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.dataset)*self.chunks_per_dataset

    def __getitem__(self, idx):
        ds = idx//self.chunks_per_dataset
        idx = idx%self.chunks_per_dataset
        return self.dataset[ds][idx]

    def stats(self):
        '''
        returns a tuple of tensors indicating the mean and standard deviation
        for each channel (I and Q)
        '''
        mean = torch.tensor([0, 0])
        var = torch.tensor([0, 0])
        n = 0
        for chunk, label in self:
            m = chunk.mean(dim=1)
            v = chunk.var(dim=1)

            nm = mean*n/(n+self.chunk_size)
            nm += m*self.chunk_size/(n+self.chunk_size)

            nv = var*(n-1)/(n+self.chunk_size-1)
            nv += torch.pow(mean, 2)*n/(n+self.chunk_size-1)
            nv += v*(self.chunk_size-1)/(n+self.chunk_size-1)
            nv += torch.pow(m, 2)*self.chunk_size/(n+self.chunk_size-1)
            nv -= torch.pow(nm, 2)*(n+self.chunk_size)/(n+self.chunk_size-1)

            mean = nm
            var = nv
            n += self.chunk_size

        return (mean, var.sqrt())

    def normalize(self, mean, std):
        '''
        mean and std are expected to be torch tensors (as returned from
        stats())
        '''
        for d in self.dataset:
            d.normalize(mean, std)


if __name__ == "__main__":
    '''
    I/Q sample rate is 20M. For a chunk of 1ms we take chunk_size of 20K.
    In a minute we collect 20M*60 = 1.2G samples (total of 1.2G*2*4=9.6GB)
    num of chunks in a minute = 1.2G/20k = 60k
    '''
    d = IQData("clear.bin", label='clear', chunk_size=20000, chunk_num=60000)
    print(d[0])
