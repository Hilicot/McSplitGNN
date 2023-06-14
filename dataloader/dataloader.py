from __future__ import annotations
import os
import torch
from torch.utils.data import Dataset
from dataloader.GraphManager import GraphManager, GraphPair
from dataloader.SearchData import SearchData, SearchDataW
import logging
from typing import List
from torch_geometric.data import Data
from options import opt
from typing import Tuple
import numpy as np
import pickle


class VDataset(Dataset):
    graph_manager: GraphManager
    search_data: List[SearchData]

    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__()
        self.graph_manager = _graph_manager
        self.search_data = []
        self.skipped_folders = 0

        # read all graph pairs
        for i, folder in enumerate(os.listdir(dataset_folder)):
            if 0 < opt.max_train_graphs <= i - self.skipped_folders:
                break
            logging.debug("Reading folder: " + folder)
            folder_path = os.path.join(dataset_folder, folder)

            if not self.check_binary_size(folder_path):
                self.skipped_folders += 1
                logging.debug(f"Skipping folder {folder} due to binaries' size")
                continue

            graph_pair = self.graph_manager.read_graph_pair(folder_path)
            self.read_from_binary_file(folder_path, graph_pair)

    def __len__(self):
        return len(self.search_data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        search_data_item = self.search_data[index]
        if opt.train_on_heuristic:
            scores = search_data_item.labels.flatten()*search_data_item.heuristics
        else:
            scores = search_data_item.labels.flatten()*search_data_item.scores
        return search_data_item.v_data.to(opt.device), scores.to(opt.device)

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        # if we already saved the search data, load it and exit
        if os.path.exists(os.path.join(folder, "v_search_data.pkl")):
            logging.debug("Reading search data from pickle")
            with open(os.path.join(folder, "v_search_data.pkl"), "rb") as f:
                local_search_data = pickle.load(f)
                self.search_data = extend_search_data_unique(self.search_data, local_search_data)
                if opt.use_diff_gnn:
                    for data in local_search_data:
                        data.graph_pair = graph_pair
                return

        # read all search data
        path = os.path.join(folder, "v_data")
        local_search_data = []
        with open(path, "rb") as f:
            # repeat until the end of the file
            i = 0
            while True:
                if opt.limit_train_data and i > opt.limit_train_data:
                    logging.warning("We are not reading all the data")
                    break
                data = SearchData(f, graph_pair)
                if not data.is_valid:
                    break
                if not data.skip:
                    local_search_data.append(data)
                    i += 1
                    if i%10000 == 0:
                        logging.debug("Reading search data: " + str(i))
        self.search_data.extend(local_search_data)

        # save search data to folder as pickle
        with open(os.path.join(folder, "v_search_data.pkl"), "wb") as f:
            logging.debug("Saving search data to pickle")
            pickle.dump(local_search_data, f)

        # resave the graph pair inside the searchData
        if opt.use_diff_gnn:
            for data in local_search_data:
                data.graph_pair = graph_pair

    def check_binary_size(self, folder):
        limit = 768*1024*1024
        return os.path.getsize(os.path.join(folder, "v_data")) <= limit and os.path.getsize(
            os.path.join(folder, "w_data")) <= limit

class WDataset(VDataset):
    def __init__(self, dataset_folder: str, _graph_manager: GraphManager):
        super().__init__(dataset_folder, _graph_manager)

    def __getitem__(self, item) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        s = self.search_data[item]
        if opt.use_diff_gnn: # use diffGNN
            labels = np.zeros((len(s.v_vertex_mapping), len(s.w_vertex_mapping)))
            # for each remapped vertex v
            for v, v_index in s.v_vertex_mapping.items():
                # for each remapped vertex w, get if it is matched to v in the solution
                for w, w_index in s.w_vertex_mapping.items():
                    if np.any(np.all(s.graph_pair.solution == [v,w], axis=1)): # if v and w are matched in the solution
                        labels[v_index][w_index] = 1
            return s.v_data.to(opt.device), s.w_data.to(opt.device), torch.tensor(labels).to(opt.device)

        # don't use diffGNN
        if opt.train_on_heuristic:
            scores = s.labels.flatten()*s.heuristics.astype(float)
        else:
            scores = s.labels.flatten()*s.scores
        return s.w_data.to(opt.device), scores.type(torch.float).to(opt.device)

    def read_from_binary_file(self, folder, graph_pair: GraphPair):
        # if we already saved the search data, load it and exit
        cache_file = os.path.join(folder, "w_search_data.pkl")
        if opt.use_diff_gnn:
            cache_file = os.path.join(folder, "w_search_data_diff.pkl")
        if os.path.exists(cache_file):
            logging.debug("Reading search data from pickle")
            with open(cache_file, "rb") as f:
                local_search_data = pickle.load(f)
                self.search_data = extend_search_data_unique(self.search_data, local_search_data)
                # resave the graph pair inside the searchData
                if opt.use_diff_gnn:
                    for data in local_search_data:
                        data.graph_pair = graph_pair
                return

        # read all search data
        path = os.path.join(folder, "w_data")
        local_search_data = []
        with open(path, "rb") as f:
            # repeat until the end of the file
            i = 0
            while True:
                if opt.limit_train_data and i > opt.limit_train_data:
                    logging.warning("We are not reading all the data")
                    break
                data = SearchDataW(f, graph_pair)
                if not data.is_valid:
                    break
                if not data.skip:
                    local_search_data.append(data)
                i += 1
                if i%10000 == 0:
                    logging.debug("Reading search data: " + str(i))
        self.search_data.extend(local_search_data)

        # save search data to folder as pickle
        with open(cache_file, "wb") as f:
            logging.debug("Saving search data to pickle")
            pickle.dump(local_search_data, f)

        # resave the graph pair inside the searchData
        if opt.use_diff_gnn:
            for data in local_search_data:
                data.graph_pair = graph_pair


def extend_search_data_unique(search_data, new_search_data):
    """
    Extends the search data with new search data, but only if the search data is not already in the new search data, else only add the new elements
    :param search_data:
    :param new_search_data:
    :return:
    """
    if len(new_search_data)  > len(search_data):
        index = 0
        for element_search, element_new in zip(search_data, new_search_data):
            if element_search != element_new:
                break
            index += 1
        search_data.extend(new_search_data[index:])
    else:
        search_data.extend(new_search_data)
    return search_data
