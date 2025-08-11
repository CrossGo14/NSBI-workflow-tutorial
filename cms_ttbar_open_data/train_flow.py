import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import zuko
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import argparse

from nsbi_common_utils.flows import EarlyStopper, CustomDataset, plot_loss, evaluate_flow

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train normalizing flows")
    parser.add_argument("--process", "-p", type=str, choices=["ttbar", "wjets", "single_top_t_chan"], required=True, help="Process to train on")
    args = parser.parse_args()

    process = args.process

    # check with torch if on GPU
    print("CUDA available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    saved_data='./cached_data/'
    dataset = pd.read_hdf(saved_data + "dataset_preselected_nominal_ttbar.h5", "dataset")

    dataset = dataset[dataset["type"] == process]
    print(len(dataset))

    # keep only 30% of events if process is ttbar
    if process == "ttbar":
        dataset = dataset.sample(frac=0.3, random_state=42)

    columns = ['log_lepton_pt', 'log_H_T', 'lepton_eta', 'lepton_phi']

    # scale to (-5, 5) where the spline is defined
    for column in columns:
        scaler = MinMaxScaler(feature_range=(-5, 5))
        dataset[column] = scaler.fit_transform(dataset[[column]]).flatten()

    flow = zuko.flows.NSF(features=len(columns), context=0, transforms=3)
    flow.to(device)

    epochs = 400
    #early_stopper = EarlyStopper(patience = 15, min_delta=0.000)
    batch_size = 256
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_dataset, columns, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDataset(val_dataset, columns, device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = CustomDataset(test_dataset, columns, device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_history = []
    test_history = []
    for epoch in range(epochs):
        start = time.time()
        print(f"Epoch {epoch}/{epochs}")
        print("Training...")
        train_losses, test_losses = [], []
        for i, (data, weights) in enumerate(train_loader):
            flow.train()
            optimizer.zero_grad()
            loss = -flow().log_prob(data)# * weights
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        train_history.append(np.mean(train_losses))

        print("Validating...")
        for i, (data, weights) in enumerate(val_loader):
            with torch.no_grad():
                flow.eval()
                loss = -flow().log_prob(data)# * weights
                loss = loss.mean()
                test_losses.append(loss.item())
        test_history.append(np.mean(test_losses))

        duration = time.time() - start
        print(
            f"Epoch {epoch} | Rank {device} - train loss: {train_history[epoch]:.4f} - val loss: {test_history[epoch]:.4f} - time: {duration:.2f}s"
        )
        #if early_stopper.early_stop(test_history[epoch]) or epoch == epochs - 1:
        if epoch == epochs - 1:
            print("Early stopping")
            # save the model
            # make dir if does not exist
            if not os.path.exists("flows/"):
                os.makedirs("flows/") 
            torch.save(flow.state_dict(), f"flows/flow_{process}.pt")
            plot_loss(train_history, test_history, "flows/", process)
            evaluate_flow(flow, test_loader, "flows/", columns, process)
            break