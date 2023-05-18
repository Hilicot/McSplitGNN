import os.path

from options import opt
from src.mcsplit import mcsplit
import logging
from dataloader.dataloader import *

def debug():
    logging.info(opt)

    folder = os.path.realpath(opt.data_folder)
    graph_manager = GraphManager()
    v_dataset = VDataset(folder, graph_manager)
    w_dataset = WDataset(folder, graph_manager)

    # TODO continue here
"""
    #### V dataset
    # split in test+train
    train_size = int(opt.train_ratio * len(v_dataset))
    test_size = len(v_dataset) - train_size
    v_train_dataset, v_test_dataset = torch.utils.data.random_split(v_dataset, [train_size, test_size])
    
    # get dataloader
    v_train_dataloader = torch.utils.data.DataLoader(v_train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    """

    #mcsplit()


if __name__ == "__main__":
    debug()