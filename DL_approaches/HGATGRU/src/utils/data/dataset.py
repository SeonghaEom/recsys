import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    # print(lens)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    # print(session_idx)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    # print(label_idx)
    idx = np.column_stack((session_idx, label_idx))
    # print(idx)
    return idx


def read_sessions(filepath):

    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    print("session shape ", sessions.shape )
    return sessions


def read_dataset(dataset_dir):
    train_sessions=read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    testf_sessions = read_sessions(dataset_dir / 'test_final.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.read())
    return train_sessions, test_sessions, testf_sessions, num_items

class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=False):
        self.sessions = sessions
        # self.graphs = graphs
        index = create_index(sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        
        return seq, label #,seq

    def __len__(self):
        return len(self.index)
