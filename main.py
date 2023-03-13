import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# This is a case to reproduce the example using LSTM to predict the RUL in s2s regression way
# Noticing that the LSTM model is developed based on pytorch


root_path = 'E:\Study\Manuscript_Intelligent maintenance\data\CMSSS'
data_path = []

for roots, dirs, files in os.walk(root_path):
    for file in files:
        if np.char.count(file, 'FD001') == 1:
            data_path.append(file)

training_data = np.loadtxt(os.path.join(roots, data_path[2]))
label = np.loadtxt(os.path.join(roots, data_path[0]))
testing_data = np.loadtxt(os.path.join(roots, data_path[1]))


def processTurboFanDataTrain(data):
    numObservations = int(np.max(data[:, 0]))
    predictors = []
    responses = []
    for i in range(numObservations):
        I = np.where(data[:, 0] == float(i) + 1.0)
        predictors.append(data[I, 2::].squeeze())
        time_steps = data[I, 1]
        responses.append(np.fliplr(time_steps).squeeze())
    return predictors, responses


XTrain, YTrain = processTurboFanDataTrain(training_data)
# Remove the constant variables, the indicators are [2,  3,  7, 12, 18, 20, 21]

In = []
for idx in range(len(XTrain)):
    x = XTrain[idx]
    m = np.min(x, 0)
    M = np.max(x, 0)
    # IdxConstant = np.where(m == M)[0]
    # In.append(IdxConstant)
    IdxConstant = [2, 3, 7, 12, 18, 20, 21]
    x = np.delete(x, IdxConstant, axis=1)
    XTrain[idx] = x

for xtrian in XTrain:
    print(xtrian.shape[1])

# Normalize the training data
for idx in range(len(XTrain)):
    x = XTrain[idx]
    mu = np.mean(x, axis=0)
    sig = np.std(x, axis=0)
    x = (x - mu) / (sig + np.finfo(float).eps)
    XTrain[idx] = x
# Clip the responses
thr = 150
for idx in range(len(YTrain)):
    y = YTrain[idx]
    y[np.where(y >= thr)[0]] = thr
    YTrain[idx] = y

# visualize the first observations and RUL
plt.subplot(2, 1, 1)
plt.plot(XTrain[0][:, 1])
plt.subplot(2, 1, 2)
plt.plot(YTrain[0])
plt.show()

# Sort the training data following the length of the data for minimum padding
# And then to evenly divide the minibatch size of the data
sequenceLengths = []
for idx in range(len(XTrain)):
    x = XTrain[idx]
    x_size = x.shape[0]
    sequenceLengths.append(int(x_size))

idx = np.argsort(sequenceLengths)
idx = idx[::-1]
xTrain = []
yTrain = []
for i in idx:
    xTrain.append(XTrain[i])
    yTrain.append(YTrain[i])
XTrain = xTrain
YTrain = yTrain

xTrain = XTrain

sequenceLengths = []
for idx in range(len(XTrain)):
    x = XTrain[idx]
    x_size = x.shape[0]
    sequenceLengths.append(int(x_size))

plt.bar(np.arange(len(sequenceLengths)), np.array(sequenceLengths))
plt.show()
# Using constant time-window to set up dataset
time_windows = 20  # Each sample includes 35 time-series features
x = []
y = []
for idx in range(len(XTrain)):
    xtrain = XTrain[idx]
    ytrain = YTrain[idx]
    samples = np.arange(0, xtrain.shape[0] - time_windows, time_windows)  # ignore the end point in some case
    for j in samples:
        x.append(xtrain[j:j + time_windows, :])
        y.append(ytrain[j:j + time_windows])
XTrain = np.array(x)
YTrain = np.array(y)
# Set up validation data from the training data
ratio4val = 0.15
num4val = int(np.around(XTrain.shape[0] * ratio4val))

indicator4val = np.random.choice(XTrain.shape[0], num4val, replace=False)
indicator4train = np.delete(np.arange(0, XTrain.shape[0]), indicator4val)
XValidation = XTrain[indicator4val, :, :]
YValidation = YTrain[indicator4val]
XTrain = XTrain[indicator4train, :, :]
YTrain = YTrain[indicator4train]


# Define a LSTM network based on Pytorch framework
miniBatchSize = 20
numResponses = 1
numHiddenUnits = 200
numFeatures = 17  # It depends on the size of filtered features in self.cnn. in this case, 5 for CNN on, 17 for LSTM only
maxEpochs = 60
learning_rate = 1e-2
num_Layers = 1


class data_Process(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor(self.Y[idx])


train_dataset = data_Process(XTrain, YTrain)
validation_dataset = data_Process(XValidation, YValidation)
train_loader = DataLoader(train_dataset, batch_size=miniBatchSize, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=miniBatchSize, shuffle=False)


class LSTM(nn.Module):
    def __init__(self, num_Features, num_Responses, num_HiddenUnits, numLayers):
        super(LSTM, self).__init__()
        self.num_Features = num_Features
        self.num_Responses = num_Responses
        self.num_HiddenUnits = num_HiddenUnits
        self.numLayer = numLayers
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=16, bias=False, kernel_size=5, stride=2),
        #     nn.BatchNorm1d(num_features=16),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=16, out_channels=32, bias=False, kernel_size=3, stride=1),
        #     nn.BatchNorm1d(num_features=32),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=32, out_channels=1, bias=False, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(num_features=1),
        # )
        self.lstm = nn.LSTM(input_size=num_Features, batch_first=True, hidden_size=num_HiddenUnits,
                            num_layers=numLayers)
        self.fc_1 = nn.Linear(num_HiddenUnits, 50)
        self.dropout = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(50, 1)

    def forward(self, inputs):
        # y_ = []
        # for idx in range(inputs.shape[1]):
        #     x_ = inputs[:, idx, :].unsqueeze(1)
        #     y_.append(self.cnn(x_))
        # x = torch.concat(y_, 1)
        x = inputs

        if x.device.type == 'cuda':
            h0 = torch.randn(self.numLayer, x.shape[0], self.num_HiddenUnits, device='cuda')
            c0 = torch.randn(self.numLayer, x.shape[0], self.num_HiddenUnits, device='cuda')
        else:
            h0 = torch.randn(self.numLayer, x.shape[0], self.num_HiddenUnits)
            c0 = torch.randn(self.numLayer, x.shape[0], self.num_HiddenUnits)
        output, (hc, cn) = self.lstm(x, (h0, c0))
        # output, hc = self.lstm(x, h0)
        pred = self.fc_1(output)
        pred = self.dropout(pred)
        z = self.fc_2(pred)
        return z.squeeze()


net = LSTM(num_Features=numFeatures, num_Responses=numResponses, num_HiddenUnits=numHiddenUnits, numLayers=num_Layers)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available')
device = torch.device('cuda:0' if use_cuda else "cpu")
net = net.to(device='cuda')
optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0, eps=1e-08)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1)
best_model_wts = copy.deepcopy(net.state_dict())

criterion = nn.MSELoss().to(device)
train_loss_list = []
val_loss_list = []
global_step = []
global_steps = 0
validation_score = torch.inf

for epoch in range(maxEpochs):
    for iteration, (XTrain, YTrain) in enumerate(train_loader):
        net.train()
        optimizer.zero_grad()
        XTrain, YTrain = XTrain, YTrain.float()
        XTrain, YTrain = XTrain.to(device), YTrain.to(device)
        YPred = net(XTrain)
        loss = criterion(YPred, YTrain)
        # Remove the next to lines if you don't need L1 or L2
        # R4L = torch.cat([x.view(-1) for x in net.parameters()])
        # l1_ = 0.001 * torch.norm(R4L, 1)
        loss = torch.sqrt(loss)
        train_loss_list.append(loss)
        current_iter = epoch * len(train_loader) + iteration + 1
        global_step.append(iteration)
        print('At Epoch ' + str(epoch + 1) + ' The training loss of Iteration ' + str(
            current_iter) + ' is ' + str(loss.detach().cpu().numpy()))
        loss.backward()  # retain_graph=True
        optimizer.step()
        if len(global_step) % 30 == 0:
            net.eval()
            for _, (XValidation, YValidation) in enumerate(validation_loader):
                XValidation, YValidation = XValidation.float(), YValidation.float()
                XValidation, YValidation = XValidation.to(device), YValidation.to(device)
                ValidationYPred = net(XValidation)
                val_loss = criterion(ValidationYPred, YValidation)
                val_loss = torch.sqrt(val_loss)
                val_loss_list.append(val_loss)
                if val_loss < validation_score:
                    validation_score = val_loss
                    best_model_wts = copy.deepcopy(net.state_dict())
    scheduler.step(loss)

training_loss = [loss.cpu().detach().numpy() for loss in train_loss_list]
plt.plot(training_loss)
plt.show()

# Test the model
net.load_state_dict(best_model_wts)
# Turn the net from gpu to cpu
net = net.to(device='cpu')


def processTurboFanDataTest(Predictors, Responses):
    predictors, _ = processTurboFanDataTrain(Predictors)
    RULTest = Responses
    numObservations = Responses.shape[0]
    responses = []
    for idx in range(numObservations):
        X = predictors[idx]
        sequenceLength = X.shape[0]
        rul = RULTest[idx]
        RUL = np.arange(rul, rul + sequenceLength)[::-1]
        responses.append(RUL)
    return predictors, responses


XTest, YTest = processTurboFanDataTest(testing_data, label)

In = []
for idx in range(len(XTest)):
    x = XTest[idx]
    m = np.min(x, 0)
    M = np.max(x, 0)
    # IdxConstant = np.where(m == M)[0]
    # In.append(IdxConstant)
    IdxConstant = [2, 3, 7, 12, 18, 20, 21]
    x = np.delete(x, IdxConstant, axis=1)
    XTest[idx] = x

# Normalize the training data
for idx in range(len(XTest)):
    x = XTest[idx]
    mu = np.mean(x, axis=0)
    sig = np.std(x, axis=0)
    x = (x - mu) / (sig + np.finfo(float).eps)
    XTest[idx] = x
    y = YTest[idx]
    y[np.where(y > thr)[0]] = thr
    YTest[idx] = y


# plot the RUL prediction
num4plt = np.random.choice(len(XTest), 16, replace=False)
for idx in range(len(num4plt)):
    indicators = num4plt[idx]
    xtest = torch.Tensor(XTest[indicators]).unsqueeze(0)
    YPred = net(xtest)
    plt.subplot(4, 4, idx+1)
    plt.plot(YPred.detach().numpy())
    plt.plot(YTest[indicators])
plt.show()
