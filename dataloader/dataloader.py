from __future__ import annotations
import os
from torch.utils.data import Dataset
from dataloader.GraphManager import GraphManager
from dataloader.SearchData import SearchData, SearchDataW

class VDataset(Dataset):
    graph_manager: GraphManager
    search_data: List[SearchData]
    def __init__(self, dataset_folder:string, _graph_manager: GraphManager):
        super().__init__()
        self.graph_manager = _graph_manager
        self.search_data = []

        # read all graph pairs
        for folder in os.listdir(dataset_folder):
            folder_path = os.path.join(dataset_folder, folder)
            graphs = lf.graph_manager.read_graph_pair(folder_path)
            self.read_from_binary_file(folder_path, graphs)


    def __len__(self):
        return len(self.search_data)

    def __getitem__(self, index):
        return self.search_data[index] # TODO add label

    def read_from_binary_file(self, folder, graphs:Tuple[Graph,Graph]):
        path = os.path.join(folder, "v_data")
        print(path)
        with open(path, "rb") as f:
            # repeat until the end of the file
            while True:
                data = SearchData(f)
                if not data.is_valid:
                    break
                data.set_graphs(graphs)
                self.search_data.append(data)


class WDataset(VDataset):
    def __init__(self, dataset_folder:string, _graph_manager: GraphManager):
        super().__init__(dataset_folder, _graph_manager)

    def read_from_binary_file(self, folder, graphs:Tuple[Graph,Graph]):
        path = os.path.join(folder, "w_data")
        print(path)
        with open(path, "rb") as f:
            # repeat until the end of the file
            while True:
                data = SearchDataW(f)
                if not data.is_valid:
                    break
                data.set_graphs(graphs)
                self.search_data.append(data)

