from options import opt
from src.mcsplit import mcsplit
import logging
import torch
from model import VGNN
import os


def test():
    v_model = None
    w_model = None

    # load models
    if opt.use_gnn_for_v:
        logging.debug("Loading V model")
        v_model = VGNN()
        v_model.load_state_dict(torch.load(os.path.join(opt.model_folder, "VGNN.pt")))
    if opt.use_gnn_for_w:
        logging.debug("Loading W model")
        w_model = VGNN()
        w_model.load_state_dict(torch.load(os.path.join(opt.model_folder, "WGNN.pt")))

    # run mcsplit
    logging.debug("Running mcsplit")
    mcsplit(v_model, w_model)


if __name__ == '__main__':
    test()
