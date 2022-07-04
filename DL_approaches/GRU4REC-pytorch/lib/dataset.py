from re import I
from xml.sax.handler import property_declaration_handler
import pandas as pd
import numpy as np
import torch


class Dataset(object):
    def __init__(self, path, sep=',', session_key='session_id', item_key='item_id', time_key='date', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):
        # Read csv
        self.df = pd.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        # self.df_purchase = pd.read_csv(p_path, sep=sep, dtype={session_key:int, item_key:int})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:
            self.df = self.df[:n_sample]

        # Add colummn item index to data
        self.add_item_indices(itemmap=itemmap)

        # Add coluumn item idx to purchased data
        self.puritemmap = pd.read_csv('./notebooks/purcand2idx_df.csv')
        self.df= self.df.merge(self.puritemmap, on=self.item_key, how='left')
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()

    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df, df_purchase
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices (1st column:0,1,10,12,15... 2nd column: 0,1,2,3,4..)
        """
        if itemmap is None:
            # item_ids = self.df[self.item_key].unique() # type is numpy.ndarray 
            # item2idx = pd.Series(data=np.arange(len(item_ids)),
            #                      index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            # itemmap = pd.DataFrame({self.item_key: item_ids,
            #                        'item_idx': item2idx[item_ids].values})
            itemmap = pd.read_csv('./notebooks/id2idx.csv')

        self.itemmap = itemmap
        self.df = self.df.merge(self.itemmap, on=self.item_key, how='left')

    def get_click_offset(self):
        """
        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32) #(session id #, )
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum() 
        return offsets

    def order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    @property
    def items(self):
        return self.itemmap[self.item_key].unique()
    @property
    def purchased_items(self):
        return self.puritemmap[self.item_key].unique()


class DataLoader():
    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]] ## 0, 3, 5
        end = click_offsets[session_idx_arr[iters] + 1] ## 3, 5, 9, 
        sid = df.session_id.values[start]
        # print(start, end) ## 0, 3, 5
        # print(df.item_idx.values[end-1]) ## 0번째 offset으 session id -> 그 session id으 purchase item의 idx 
        # p_target = self.dataset.get_pitems(session_idx_arr[iters])
        # assert len(start) == len(end)
        # assert len(end) == len(p_target)
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                p_target = torch.LongTensor(df.pur_idx.values[end-1])

                flag = True if i == minlen-2 else False
                yield input, target, p_target, mask, sid, flag

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)


            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
