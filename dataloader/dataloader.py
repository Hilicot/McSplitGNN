from __future__ import annotations
import os
from torch.utils.data import Dataset
from dataloader.GraphManager import GraphManager, GraphPair
from dataloader.SearchData import SearchData, SearchDataW
import logging
from typing import List
from torch_geometric.data import Data


class VDataset(Dataset):
    graph_manager: GraphManager
    search_data: List[SearchData]

    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__()
        self.graph_manager = _graph_manager
        self.search_data = []

        # read all graph pairs
        for folder in os.listdir(dataset_folder):
            logging.debug("Reading folder: " + folder)
            folder_path = os.path.join(dataset_folder, folder)
            graph_pair = self.graph_manager.read_graph_pair(folder_path)
            self.read_from_binary_file(folder_path, graph_pair)

    def __len__(self):
        return len(self.search_data)

    def __getitem__(self, index) -> Data:
        search_data_item = self.search_data[index]
        return search_data_item.data

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        path = os.path.join(folder, "v_data")
        with open(path, "rb") as f:
            # repeat until the end of the file
            while True:
                data = SearchData(f, graph_pair)
                if not data.is_valid:
                    break
                self.search_data.append(data)


class WDataset(VDataset):
    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__(dataset_folder, _graph_manager)

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        path = os.path.join(folder, "w_data")
        with open(path, "rb") as f:
            # repeat until the end of the file
            while True:
                data = SearchDataW(f, graph_pair)
                if not data.is_valid:
                    break
                self.search_data.append(data)
