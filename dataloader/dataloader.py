from __future__ import annotations
import os
from torch.utils.data import Dataset
from dataloader.GraphManager import GraphManager, GraphPair
from dataloader.SearchData import SearchData, SearchDataW
import logging
from typing import List
from torch_geometric.data import Data
from options import opt
from typing import Tuple
import numpy as np


class VDataset(Dataset):
    graph_manager: GraphManager
    search_data: List[SearchData]

    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__()
        self.graph_manager = _graph_manager
        self.search_data = []

        # read all graph pairs
        for i, folder in enumerate(os.listdir(dataset_folder)):
            if 0 < opt.max_train_graphs <= i:
                break
            logging.debug("Reading folder: " + folder)
            folder_path = os.path.join(dataset_folder, folder)
            graph_pair = self.graph_manager.read_graph_pair(folder_path)
            self.read_from_binary_file(folder_path, graph_pair)

    def __len__(self):
        return len(self.search_data)

    def __getitem__(self, index) -> Tuple[Data, np.ndarray]:
        search_data_item = self.search_data[index]
        return search_data_item.v_data, search_data_item.labels

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        path = os.path.join(folder, "v_data")
        with open(path, "rb") as f:
            # repeat until the end of the file
            i = 0
            while True:
                if i > 100 and False:
                    logging.warning("We are not reading all the data")
                    break
                data = SearchData(f, graph_pair)
                if not data.is_valid:
                    break
                if not data.skip:
                    self.search_data.append(data)
                    i+=1


class WDataset(VDataset):
    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__(dataset_folder, _graph_manager)

    def __getitem__(self, item) -> Tuple[Data, Data, np.ndarray]:
        search_data_item = self.search_data[item]
        return search_data_item.w_data, search_data_item.labels

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        path = os.path.join(folder, "w_data")
        with open(path, "rb") as f:
            # repeat until the end of the file
            while True:
                data = SearchDataW(f, graph_pair)
                if not data.is_valid:
                    break
                if not data.skip:
                    self.search_data.append(data)
