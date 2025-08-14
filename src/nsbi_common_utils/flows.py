from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomDataset(Dataset):
    def __init__(self, pd_dataset, columns, device=None):
        self.data = pd_dataset[columns].values
        self.columns = columns
        #self.weights = pd_dataset["weights_normed"].values
        self.weights = pd_dataset["weights"].values
        if device is not None:
            self.data = torch.tensor(self.data, dtype=torch.float32).to(device)
            self.weights = torch.tensor(self.weights, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.weights[idx]


def plot_loss(train_history, test_history, directory, process):
    fig, ax = plt.subplots()
    ax.plot(train_history, label="Train")
    ax.plot(test_history, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(directory + f"{process}_loss.png")
    plt.close(fig)


def evaluate_flow(flow, test_loader, directory, columns, process):
    # plot distribution sampled from the flow and compare to the original distribution
    # test_dataset is a pandas
    flow = flow.to("cpu")
    with torch.no_grad():
        sample_list = []
        data_list = []
        weights_list = []
        for data, weights in test_loader:
            data = data.detach().cpu().numpy()
            weights = weights.detach().cpu().numpy()
            samples = flow().sample((len(data),))
            sample = samples.reshape(-1, len(columns))
            sample_list.append(samples)
            data_list.append(data)
            weights_list.append(weights)
            #print("Length of data: ", len(data))
            #print("Length of sample: ", len(sample))
            #print("Length of weights: ", len(weights))
    sample = np.concatenate(sample_list)
    data = np.concatenate(data_list)
    weights = np.concatenate(weights_list)

    # plot
    for i, column in enumerate(columns):
        fig, ax = plt.subplots()
        ax.hist(data[:, i], bins=100, histtype="step", label="Test data", weights=weights, density=False)
        ax.hist(sample[:, i], bins=100, histtype="step", label="Flow", weights=weights, density=False)
        ax.set_xlabel(column)
        ax.legend()
        fig.savefig(directory + f"flow_{process}_{column}.png")
        plt.close(fig)