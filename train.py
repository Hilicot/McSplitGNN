import os.path

from typing import Tuple
from options import opt
from src.mcsplit import mcsplit
import logging
from dataloader.dataloader import *
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from model.VGNN import VGNN
from model.WGNN import WGNN
from torch.optim import Adam
from torch.nn import BCELoss, MSELoss
import torch
import numpy as np

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from datetime import datetime


def train():
    data_types = [
        "V",
        "W"
    ]

    graph_manager = GraphManager()

    if "V" in data_types:
        train_model("V", VDataset, VGNN, graph_manager)
    if "W" in data_types:
        train_model("W", WDataset, VGNN, graph_manager)

    # mcsplit()


def train_model(data_type, dataset_t, model_t, graph_manager):
    logging.info("Training {}".format(data_type))

    # init dataset
    folder = os.path.realpath(opt.data_folder)

    dataset = dataset_t(folder, graph_manager)
    train_size = int(opt.train_ratio/100*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    # init model
    criterion = MSELoss()
    model = model_t()
    optimizer = Adam(model.parameters(), lr=0.005)

    # train model
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    for epoch in range(opt.epochs):
        model.train()
        avg_train_loss, avg_train_acc = run_epoch(True, train_dataloader, model, criterion, optimizer)
        model.eval()
        avg_test_loss, avg_test_acc = run_epoch(False, test_dataloader, model, criterion, optimizer)

        logging.info(
            "{},\tEpoch {},\tTrain_Loss {},\tTrain_Accuracy {},\tTest_Loss {},\tTest_Accuracy {}".format(data_type,
                                                                                                         epoch,
                                                                                                         avg_train_loss,
                                                                                                         avg_train_acc,
                                                                                                         avg_test_loss,
                                                                                                         avg_test_acc))
        train_losses.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        test_losses.append(avg_test_loss)
        test_acc.append(avg_test_acc)

    # plot losses
    plot_losses("Train", train_losses, train_acc)
    plot_losses("Test", test_losses, test_acc)

    # save model
    if opt.save_model:
        logging.debug("Saving model")
        folder = os.path.join(os.path.realpath(opt.model_folder), f"graphs_{opt.max_train_graphs}_epochs_{opt.epochs}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        if data_type == "V":
            torch.save(model.state_dict(), os.path.join(folder, "VGNN.pt"))
        elif data_type == "W":
            torch.save(model.state_dict(), os.path.join(folder, "WGNN.pt"))


def run_epoch(is_train: bool, dataloader, model, criterion, optimizer) -> Tuple[float, float]:
    total_loss = 0
    total_pred = 0

    for data in dataloader:
        # Forward pass
        optimizer.zero_grad()
        output = model(data[0])

        # Compute loss
        assert len(data[-1]) == 1
        label = data[-1][0]
        loss = criterion(output, label)
        pred = check_predition(output, label)

        # Backpropagation and optimization
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_pred += pred

    avg_loss = total_loss/len(dataloader)
    accuracy = total_pred/len(dataloader)
    return avg_loss, accuracy


def plot_losses(phase, losses, accuracy=None):
    epochs = range(1, len(losses) + 1)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plot the loss graph
    plt.plot(epochs, losses, 'b', label=f'{phase}ing Loss')
    plt.title(f'{phase}ing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if opt.log_folder:
        plt.savefig(os.path.join(opt.log_folder, f"{opt.timestamp}_{phase}_loss.png"))
    plt.show()

    if accuracy is not None:
        # Plot the accuracy graph
        plt.plot(epochs, accuracy, 'r', label=f'{phase}ing Accuracy')
        plt.title(f'{phase}ing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if opt.log_folder:
            plt.savefig(os.path.join(opt.log_folder, f"{opt.timestamp}_{phase}_accuracy.png"))
        plt.show()


def check_predition(output, label):
    return 1 if torch.argmax(torch.flatten(output)) == torch.argmax(torch.flatten(label)) else 0


if __name__ == "__main__":
    train()
