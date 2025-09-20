from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_data = len(data_source)
        self.num_pids_per_batch = self.batch_size

        self.index_dic = {}
        for index, (pid,_, _, _) in enumerate(self.data_source):
            if pid in self.index_dic:
                self.index_dic[pid].append(index)
            else:
                self.index_dic[pid] = [index]

        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        pids_copy = copy.deepcopy(self.pids[:])
        final_idxs = []
        index_dic = copy.deepcopy(self.index_dic)
        num_data = copy.deepcopy(self.num_data)

        while (num_data-len(final_idxs)) >= self.num_pids_per_batch or len(final_idxs)%self.num_pids_per_batch != 0:
            selected_pids = random.sample(pids_copy, self.num_pids_per_batch)
            for pid in selected_pids:
                if len(index_dic[pid]) > 0:
                    idxs = random.sample(index_dic[pid], 1)
                    final_idxs.extend(idxs)
                    index_dic[pid].remove(idxs[0])
                if len(final_idxs) % self.num_pids_per_batch == 0:
                    break

        return iter(final_idxs)

    def __len__(self):
        return len(self.data_source)