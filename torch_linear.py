import torch
import numpy as np
from torch import nn
from torch import optim
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from utils import load_dataset, concat_train_valid, print_metrics


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(7800, 1)
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.zeros_(self.layer1.bias)
        self.model = nn.Sequential(
            nn.Linear(520, 520),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(520, 260),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(260, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x):
        out = torch.sigmoid(self.model(x))
        return out


class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class TestDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.length


def binary_acc(preds, truth):
    pred_tags = torch.round(preds)
    correct_results_sum = (pred_tags == truth).sum().float()
    accuracy = correct_results_sum / truth.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy


# def binary_roc_auc(preds, truth):
#     preds = preds.detach().cpu().numpy()
#     truth = truth.detach().cpu().numpy()
#     roc_auc = metrics.roc_auc_score(truth, preds)
#     return roc_auc


def make_preds(model, data_loader):
    preds = list()
    model.eval()
    with torch.no_grad():
        for x_test_batch in data_loader:
            x_test_batch = x_test_batch.to(device)
            y_test_pred = torch.round(model(x_test_batch))
            preds.append(y_test_pred.cpu().numpy().item())
    preds = np.array(preds, dtype=np.uint8).reshape((-1, 1))
    return preds


if __name__ == '__main__':
    # Load dataset
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    X_train, Y_train = concat_train_valid(V, Y)
    X_test = V['test']
    Y_test = Y['test']
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    train_set = TrainDataset(X_train, Y_train)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = TestDataset(X_test)
    test_loader = DataLoader(test_set, batch_size=1)

    # Create model
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    network = Network().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.1)

    # Train model
    for e in range(100):
        epoch_loss = 0
        epoch_acc = 0
        # epoch_roc_auc = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_preds = network(x_batch)
            loss = criterion(y_preds, y_batch)
            acc = binary_acc(y_preds, y_batch)
            # roc_auc = binary_roc_auc(y_preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # epoch_roc_auc += roc_auc.item()
        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | '
            f'Acc: {epoch_acc / len(train_loader):.3f}')

    # Test model
    model_preds = make_preds(network, test_loader)
    print_metrics(model_preds, Y_test)
