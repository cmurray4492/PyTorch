import copy
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import get_time_series_datasets
from dummy_model import DummyPredictor
from fcnn_model import FCNN
from hwes_model import HwesPredictor
from linear_interpolation_model import INterpolationPredictor

random.seed(1)
torch.manual_seed(1)

features = 1
ts_len = 3000

x_train, x_val, x_test, y_train, y_val, y_test = get_time_series_datasets(features, ts_len)

net = FCNN(n_inp=features, l_1=64, l_2=32, n_out=1)
net.train()

dummy_predictor = DummyPredictor()
interpolation_predictor = INterpolationPredictor()
hwes_predictor = HwesPredictor()

optimizer = torch.optim.Adam(params=net.parameters())
loss_func = torch.nn.MSELoss()

best_model = None
min_val_loss = 1000000
training_loss = []
validation_loss = []

for t in range(10000):
    prediction = net(x_train)
    loss = loss_func(prediction, x_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_prediction = net(x_val)
    val_loss = loss_func(val_prediction, y_val)
    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_model = copy.deepcopy(net)
        min_val_loss = val_loss.item()
    if t % 1000 == 0:
        print(f"epoch {t}: train - {round(loss.item(), 4)}, val: - {round(val_loss.item(), 4)}")

net.eval()
print('Testing')
print(f"FCNN Loss: {loss_func(best_model(x_test), y_test).item()}")
print(f"Dummy Loss: {loss_func(dummy_predictor(x_test), y_test).item()}")
# print(f"Linear Interpolation: {loss_func(interpolation_predictor(x_test), y_test).item()}")
# print(f"HWES Loss: {loss_func(hwes_predictor(x_test), y_test).item}")
