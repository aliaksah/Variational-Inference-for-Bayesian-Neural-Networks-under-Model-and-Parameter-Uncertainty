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
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(1)

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

# define the Gaussian distribution
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


# define Bernoulli distribution
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


# set prior parameters
PI = 1
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(256, 400)
        self.l2 = BayesianLinear(400, 600)
        self.l3 = BayesianLinear(600, 5)

    def forward(self, x, sample=False):
        x = x.view(-1, 256)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


net = BayesianNetwork().to(DEVICE)


def write_weight_histograms(epoch, i):
    aaa = 5


def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    aaa = 5


def train(net, optimizer, epoch, i):
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
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
    print(epoch + 1)
    print(loss)
    print(negative_log_likelihood)


def test_ensemble():
    net.eval()
    correct = 0

    correct3 = 0
    cases3 = 0


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

            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)

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

            outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs[0:TEST_SAMPLES].mean(0)
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


print("Classes loaded")

# %%

# make inference on 10 networks
for i in range(0, 10):
    print(i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam([
        {'params': net.l1.bias_mu, 'lr': 0.0001},
        {'params': net.l2.bias_mu, 'lr': 0.0001},
        {'params': net.l3.bias_mu, 'lr': 0.0001},
        {'params': net.l1.bias_rho, 'lr': 0.0001},
        {'params': net.l2.bias_rho, 'lr': 0.0001},
        {'params': net.l3.bias_rho, 'lr': 0.0001},
        {'params': net.l1.weight_mu, 'lr': 0.0001},
        {'params': net.l2.weight_mu, 'lr': 0.0001},
        {'params': net.l3.weight_mu, 'lr': 0.0001},
        {'params': net.l1.weight_rho, 'lr': 0.0001},
        {'params': net.l2.weight_rho, 'lr': 0.0001},
        {'params': net.l3.weight_rho, 'lr': 0.0001},
    ], lr=0.0001)
    for epoch in range(epochs):
        train(net, optimizer, epoch, i)

    res = test_ensemble()

    np.savetxt("soundGmaccuracies_" + str(i) + ".csv", res, delimiter=",")

# %%

#res = test_ensemble()
#np.savetxt("soundGmaccuracies_" + str(i) + ".csv", res, delimiter=",")

# %%/mn/sarpanitu/ansatte-u2/aliaksah/PycharmProjects/bnn2/sound_vanilla_BNN.py
