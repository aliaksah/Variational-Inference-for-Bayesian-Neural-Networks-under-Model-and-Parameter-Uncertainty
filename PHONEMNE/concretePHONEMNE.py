#matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import pandas as pd

prefix = "_phoneme_bg_"
# define the summary writer
writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(3)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
batch_size = 100
COND_OPT = False
CLASSES = 5
# TRAIN_EPOCHS = 250
SAMPLES = 1
TEST_SAMPLES = 10
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 250
pepochs = 50

#prepare the data
data = pd.read_csv('http://www.uio.no/studier/emner/matnat/math/STK2100/data/phoneme.data')
data = data.drop(columns=["row.names"])
data = pd.concat([data,data.g.astype("category").cat.codes.astype(int)],sort=False, axis=1) #get_dummies(data['g'], prefix='phoneme')],sort=False, axis=1)
data = data.drop(columns=["g","speaker"])
data = data.values

np.random.seed(40590)
tr_ids = np.random.choice(4509, 3500, replace = False)
te_ids = np.setdiff1d(np.arange(4509),tr_ids)[0:1000]

dtrain = data[tr_ids,:]

data_mean = dtrain.mean(axis=0)[0:256]
data_std = dtrain.std(axis=0)[0:256]

data[:,0:256] = (data[:,0:256]  - data_mean)/data_std




dtrain = data[tr_ids,:]
dtest = data[te_ids,:]


TRAIN_SIZE = len(tr_ids)
TEST_SIZE = len(te_ids)
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE
NUM_TEST_BATCHES = len(te_ids)/BATCH_SIZE


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.9):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        # print(input_dimensionality)
        out = (self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        # print(x.size())

        return x


class Model(nn.Module):
    def __init__(self, weight_regularizer, dropout_regularizer):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(256, 400)
        self.linear2 = nn.Linear(400, 600)
        self.linear3 = nn.Linear(600, 5)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.relu = nn.ReLU()

    def forward(self, x):
        regularization = torch.empty(3, device=x.device)

        x = x.view(-1, 256)
        # print(x.size())
        x1 = self.linear1(x)
        x1, regularization[0] = self.conc_drop1(x1, self.linear1)
        x1 = self.relu(x1)
        x2 = self.linear2(x1)
        x2, regularization[1] = self.conc_drop2(x2, self.linear2)
        x2 = self.relu(x2)
        x3 = self.linear3(x2)
        x3, regularization[2] = self.conc_drop3(x3, self.linear3)
        x3 = self.relu(x3)
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        return F.log_softmax(x3, dim=1), regularization.sum()


def concrete_loss(y, y_out, regularization):
    # print(y)
    # print(torch.argmax(y_out))
    loss = F.nll_loss(y_out, y, reduction='mean') + regularization
    return loss


def train(net, optimizer, epoch, i):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:256]
        _y = dtrain[old_batch: batch_size * batch, 256:257]
        old_batch = batch_size * batch
        # print(_x.shape)
        # print(_y.shape)

        data = Variable(torch.FloatTensor(_x)).cuda()
        target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]
        net.zero_grad()
        y_fit, regularization = net.forward(data)
        loss = concrete_loss(target, y_fit, regularization)
        loss.backward()
        optimizer.step()
    print(epoch + 1)
    print(loss)


def test_ensemble():
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(dtest.shape[0] / batch_size))):
            batch = (batch + 1)
            _x = dtest[old_batch: batch_size * batch, 0:256]
            _y = dtest[old_batch: batch_size * batch, 256:257]

            old_batch = batch_size * batch

            # print(_x.shape)
            # print(_y.shape)

            data = Variable(torch.FloatTensor(_x)).cuda()
            target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]
            # print(target)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i], b = net(data)
                # print(outputs[i])
            outputs[TEST_SAMPLES], b = net(data)
            output = outputs[0:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    corrects = np.append(corrects, correct)

    return corrects


print("Classes loaded")

# %%

i = 1
torch.manual_seed(i)
wr = 1e-6
dr = 1e-5
net = Model(wr, dr).to(DEVICE)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
for epoch in range(epochs):
    train(net, optimizer, epoch, i)

# %%

torch.sigmoid(net.conc_drop1.p_logit)


# %%

def test_ensemble():
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    correct3 = 0
    cases3 = 0
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(dtest.shape[0] / batch_size))):
            batch = (batch + 1)
            _x = dtest[old_batch: batch_size * batch, 0:256]
            _y = dtest[old_batch: batch_size * batch, 256:257]

            old_batch = batch_size * batch

            # print(_x.shape)
            # print(_y.shape)

            data = Variable(torch.FloatTensor(_x)).cuda()
            target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]
            # print(target)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i], b = net(data)
                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp

            mydata_means /= TEST_SAMPLES

            outputs[TEST_SAMPLES], b = net(data)
            output = outputs.mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(mydata_means[1][1])
            for jj in range(TEST_BATCH_SIZE):
                if mydata_means[jj][pred.detach().cpu().numpy()[jj]] >= 0.95:
                    correct3 += pred[jj].eq(target.view_as(pred)[jj]).sum().item()
                    cases3 += 1

            if cases3 == 0:
                cases3 += 1
    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    corrects = np.append(corrects, correct)
    corrects = np.append(corrects, correct3 / cases3)
    corrects = np.append(corrects, cases3)
    return corrects

# test_ensemble()

    # %%

    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # print(target)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i], b = net(data)
                # print(outputs[i])
            outputs[TEST_SAMPLES], b = net(data)
            output = outputs.mean(0)
            preds = preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    corrects = np.append(corrects, correct)


# %%

# make inference on 10 networks
for i in range(0, 10):
    torch.manual_seed(i)
    wr = 1e-6
    dr = 1e-5
    net = Model(wr, dr).to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train(net, optimizer, epoch, i)

    res = test_ensemble()
    res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
                          torch.sigmoid(net.conc_drop3.p_logit)])
    np.savetxt("SoundConrDropr_" + str(i) + ".csv", res, delimiter=",")

    net.conc_drop1.p_logit.requires_grad = False
    net.conc_drop2.p_logit.requires_grad = False
    net.conc_drop3.p_logit.requires_grad = False

    for epoch in range(pepochs):
        train(net, optimizer, epoch, i)
        print(torch.sigmoid(net.conc_drop1.p_logit))

    res = test_ensemble()
    res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
                          torch.sigmoid(net.conc_drop3.p_logit)])
    np.savetxt("SoundConrDroprPt_" + str(i) + ".csv", res, delimiter=",")

# %%
