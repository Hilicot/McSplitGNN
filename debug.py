import os.path

from options import opt
from src.mcsplit import mcsplit
import logging
from dataloader.dataloader import *
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from model.VGNN import VGNN
from torch.optim import Adam
from torch.nn import MSELoss

def debug():
    logging.info(opt)

    folder = os.path.realpath(opt.data_folder)
    graph_manager = GraphManager()
    v_dataset = VDataset(folder, graph_manager)
    w_dataset = WDataset(folder, graph_manager)

    #### V dataset
    # split in test+train
    train_size = int(opt.train_ratio/100 * len(v_dataset))
    test_size = len(v_dataset) - train_size
    v_train_dataset, v_test_dataset = random_split(v_dataset, [train_size, test_size])
    
    # get dataloader
    v_train_dataloader = DataLoader(v_train_dataset, batch_size=opt.batch_size, shuffle=True)
    v_test_dataloader = DataLoader(v_test_dataset, batch_size=opt.batch_size, shuffle=True)

    # init model
    # TODO define dims
    logging.warning("TODO properly define dims")
    input_dim = 1
    hidden_dim = 16
    output_dim = 1
    model = VGNN(input_dim, hidden_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in v_train_dataloader:
            data = data.to(opt.device)

            # TODO still to validate
            """# Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Compute loss
            loss = criterion(output, data.y)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()"""

        avg_loss = total_loss/len(v_train_dataloader)
        logging.info("Epoch {}: Loss {}".format(epoch, avg_loss))

    #mcsplit()


if __name__ == "__main__":
    debug()