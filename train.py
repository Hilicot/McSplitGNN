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
from model.DiffGNN import DiffGNN
from torch.optim import Adam
from torch.nn import BCELoss, MSELoss
import torch
import numpy as np

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from datetime import datetime


def train():
    graph_manager = GraphManager()

    if opt.train_v:
        train_model("V", VDataset, VGNN, graph_manager)
    if opt.train_w:
        train_model("W", WDataset, VGNN, graph_manager)
    if opt.use_diff_gnn:
        train_model("Diff", WDataset, DiffGNN, graph_manager)

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
    model = model.to(opt.device)
    optimizer = Adam(model.parameters(), lr=0.005)

    # train model
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    logging.debug("Training {} for {} epochs".format(data_type, opt.epochs))
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

        # save model
        if opt.save_model:
            logging.debug("Saving model")
            folder = os.path.join(os.path.realpath(opt.model_folder),
                                  f"graphs_{opt.max_train_graphs}_epochs_{epoch + 1}")
            if not os.path.exists(folder):
                os.makedirs(folder)
            if data_type == "V":
                torch.save(model.state_dict(), os.path.join(folder, "VGNN.pt"))
            elif data_type == "W":
                torch.save(model.state_dict(), os.path.join(folder, "WGNN.pt"))
            elif data_type == "Diff":
                torch.save(model.state_dict(), os.path.join(folder, "DiffGNN.pt"))

    # plot losses
    plot_losses("Train", train_losses, train_acc)
    plot_losses("Test", test_losses, test_acc)


def run_epoch(is_train: bool, dataloader, model, criterion, optimizer) -> Tuple[float, float]:
    total_loss = 0
    total_pred = 0

    if not opt.use_diff_gnn:
        for i,data in enumerate(dataloader):
            data, label = data
            # Forward pass
            optimizer.zero_grad()
            output = model(data).flatten()

            # Compute loss
            assert len(label) == 1
            label = torch.sigmoid(label[0]/100)
            loss = criterion(output, label)
            pred = check_predition(output, label)

            # Backpropagation and optimization
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_pred += pred
            if i%100000 == 0 and i > 0:
                logging.debug("Batch {},\tLoss {},\tAccuracy {}".format(i, total_loss/(i + 1), total_pred/(i + 1)))

    else: # use_diff_gnn

        for i, data in enumerate(dataloader):
            v_data, w_data, label = data
            label = label[0]
            # Forward pass
            optimizer.zero_grad()
            v_embs = model(v_data)
            w_embs = model(w_data)

            loss = None
            # handle all v_embs separately for simplicity of code
            for j, v_emb in enumerate(v_embs):
                # balance w_embs delete some w_embs such that we have an equal numebr of w_embs with label 0 and 1
                w_embs_bal, w_label_bal = balanced_subset(w_embs, label[j])

                # compute target difference
                w_labels = torch.broadcast_to(w_label_bal.unsqueeze(1), (-1, 64)) # matrix. for each w_emb, we have 64 identical values 0 or 1 (= w non matched or matched to v)
                target_diffs = -(w_labels-1)/2  # if v and w are matched, diff must be 0, else 0.5
                target_diffs = target_diffs.float().to(opt.device)

                # compute diffence between embeddings (w/ broadcasting)
                diff = torch.abs(v_emb - w_embs_bal)

                # compute loss of the difference
                _loss = criterion(diff, target_diffs)
                if loss is None:
                    loss = _loss
                else:
                    loss += _loss

            # Backpropagation and optimization
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_pred = 0
            if i%100000 == 0 and i > 0:
                logging.debug("Batch {},\tLoss {}".format(i, total_loss/(i + 1)))

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
    plt.clf()

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
        plt.clf()


def check_predition(output, label):
    return 1 if torch.argmax(torch.flatten(output)) == torch.argmax(torch.flatten(label)) else 0


def balanced_subset(w_embs, w_labels):
    # Find indices of elements with label 0 and label 1
    label_0_indices = torch.where(w_labels == 0)[0]
    label_1_indices = torch.where(w_labels == 1)[0]

    # Determine the smaller count of label 0 and label 1
    subset_size = max(min(len(label_0_indices), len(label_1_indices)),1)

    # Randomly select subset_size number of indices from each label
    label_0_subset_indices = torch.randperm(len(label_0_indices))[:subset_size]
    label_1_subset_indices = torch.randperm(len(label_1_indices))[:subset_size]

    # Get the subset of w_embs based on the selected indices
    balanced_subset_indices = torch.cat([label_0_indices[label_0_subset_indices], label_1_indices[label_1_subset_indices]])
    balanced_subset = w_embs[balanced_subset_indices]
    balanced_subset_labels = w_labels[balanced_subset_indices]

    return balanced_subset, balanced_subset_labels


if __name__ == "__main__":
    train()
