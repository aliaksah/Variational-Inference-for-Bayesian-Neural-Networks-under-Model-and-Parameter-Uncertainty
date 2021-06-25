# define the summary writer

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

writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(0)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# load the data

BATCH_SIZE = 100
TEST_BATCH_SIZE = 5
COND_OPT = False

epochs = 250

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        './fmnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        './fmnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=5, shuffle=False)
TRAIN_SIZE = len(train_loader.dataset)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

CLASSES = 10
TRAIN_EPOCHS = 2
SAMPLES = 1
TEST_SAMPLES = 10

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


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
PI = 0.5  # if set to 1 or 0 no mixtures are addressed
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
        self.l1 = BayesianLinear(28 * 28, 400)
        self.l2 = BayesianLinear(400, 600)
        self.l3 = BayesianLinear(600, 10)

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
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
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction='sum')
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


net = BayesianNetwork().to(DEVICE)


def write_weight_histograms(epoch, i):
    aaa = 5


def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    aaa = 5


def train(net, optimizer, epoch, i):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
    print(epoch + 1)
    print(loss)
    print(negative_log_likelihood)


def test_ensemble(net):
    net.eval()
    correct = 0
    correct3 = 0
    cases3 = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
                if (i == 0):
                    fmnist_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        fmnist_means[j] /= np.sum(fmnist_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    fmnist_means = fmnist_means + tmp
            fmnist_means /= TEST_SAMPLES
            outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs[0:TEST_SAMPLES].mean(0)

            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            for jj in range(TEST_BATCH_SIZE):
                if fmnist_means[jj][pred.detach().cpu().numpy()[jj]] >= 0.95:
                    correct3 += pred[jj].eq(target.view_as(pred)[jj]).sum().item()
                    cases3 += 1
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
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


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def outofsample(net):
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
    # gt1 = np.zeros((400,784))
    # gt2 = np.zeros((600,400))
    # gt3 = np.zeros((10,600))
    # ots = np.zeros(obj, dtype=int)
    # dts = np.zeros((obj,5,784))
    entropies = np.zeros(10)
    count = 0
    k = 0
    spars = np.zeros(TEST_SAMPLES)
    with torch.no_grad():
        for data, target in mnist_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                # print(i)
                outputs[i] = net.forward(data, sample=True)
                if (i == 0):
                    fmnist_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        fmnist_means[j] /= np.sum(fmnist_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    fmnist_means = fmnist_means + tmp
            fmnist_means /= TEST_SAMPLES
            # print(np.sum(fmnist_means))
            for j in range(TEST_BATCH_SIZE):
                if k == 0 and j == 0:
                    entropies = -np.sum(fmnist_means[j] * np.log(fmnist_means[j]))
                else:
                    entropies = np.append(entropies, -np.sum(fmnist_means[j] * np.log(fmnist_means[j])))
            k += 1
            output = outputs[0:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    cdf(entropies.flatten())
    return entropies.flatten()


print("Classes loaded")

# %%

# make inference on nruns networks
nruns = 1
for i in range(9,10):
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
    res = test_ensemble(net)

    enr = outofsample(net)
    np.savetxt(X=enr, fname="entromixgaussfmm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "mixgaussffm" + str(i) + ".par")

    np.savetxt("mixgaussfmaccuracies_" + str(i) + ".csv", res, delimiter=",")

# %%

for i in range(nruns):
    torch.manual_seed(i)
    print(i)
    net = BayesianNetwork().to(DEVICE)
    net.load_state_dict(torch.load("mixgaussffm" + str(i) + ".par"))
    print("loaded simple")
    enr = outofsample(net)
    np.savetxt(X=enr, fname="entromixgaussfmm" + str(i) + ".csv", delimiter=",")


# %%

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


# %%

# plot in sample example of uncertainties  based on Bayesian model averaging
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


fmnist_sample = iter(test_loader).next()
fmnist_sample[0] = fmnist_sample[0].to(DEVICE)
print(fmnist_sample[1])
sns.set_style("dark")
show(make_grid(fmnist_sample[0].cpu()))
net.eval()
res = net.forward(fmnist_sample[0], sample=True)
fmnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
fmnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(fmnist_sample[0], sample=True)
    fmnist_outputs = np.append(fmnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    fmnist_means = fmnist_means + sigmoid(res.detach().cpu().numpy())
fmnist_means /= 100
# fmnist_means = fmnist_means/np.sum(np.reshape(fmnist_means,(5,10)),axis=0)
# fmnist_means = sigmoid(fmnist_means)
sns.set_style("dark")
fig, ax = plt.subplots(5, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 8]})
for j in range(5):
    if (j == 4):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10), ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    else:
        ax[j, 1].set_xticks([], [])
    ax[j, 1].set_ylabel("Count")
    ax[j, 1].set_yticks(range(50, 101, 50))
    n, bins, patches = ax[j, 1].hist(fmnist_outputs[j], np.arange(-0.5, 10, 1), color="b")
    patches[fmnist_sample[1][j]].set_fc('g')
    ax[j, 1].plot(range(10), fmnist_means[j] / np.sum(fmnist_means[j]) * 100, lw=2, color="y")
    ax[j, 1].axhline(95, color='k', linestyle='dashed', linewidth=1)
    ax[j, 0].imshow(fmnist_sample[0].cpu().numpy()[j, 0, 0:27, 0:27])
    ax[j, 0].set_xticks([], [])
    ax[j, 0].set_yticks([], [])
    ax[j, 0].set_ylabel("Truth")
plt.sca(ax[4, 1])
plt.xticks(range(10),
           ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
plt.show()

# %%

# plot out of sample (MNIST) example of uncertainties  based on Bayesian model averaging
mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=5, shuffle=False)
mnist_sample = iter(mnist_loader).next()
mnist_sample[0] = mnist_sample[0].to(DEVICE)
print(mnist_sample[1])
sns.set_style("dark")
show(make_grid(mnist_sample[0].cpu()))

net.eval()
res = net.forward(mnist_sample[0], sample=True)
mnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
mnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(mnist_sample[0], sample=True)
    mnist_outputs = np.append(mnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    mnist_means = mnist_means + sigmoid(res.detach().cpu().numpy())
mnist_means /= 100
sns.set_style("dark")
fig, ax = plt.subplots(5, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 8]})
for j in range(5):
    if (j == 4):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10), ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    else:
        ax[j, 1].set_xticks([], [])
    ax[j, 1].set_ylabel("Count")
    ax[j, 1].set_yticks(range(50, 101, 50))
    n, bins, patches = ax[j, 1].hist(mnist_outputs[j], np.arange(-0.5, 10, 1), color="b")
    ax[j, 1].plot(range(10), mnist_means[j] / np.sum(mnist_means[j]) * 100, lw=2, color="y")
    # patches[pts[i]].set_fc('r')
    # patches[fmnist_sample[1][j]].set_fc('g')
    ax[j, 1].axhline(95, color='k', linestyle='dashed', linewidth=1)
    ax[j, 0].imshow(mnist_sample[0].cpu().numpy()[j, 0, 0:27, 0:27])
    ax[j, 0].set_xticks([], [])
    ax[j, 0].set_yticks([], [])
    ax[j, 0].set_ylabel("Truth")
plt.sca(ax[4, 1])
# the closer to uniform distribution the better
plt.xticks(range(10),
           ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
plt.show()

# %%

# analyze the in-domain results (FMNIST)
net.eval()
correct = 0
obj = 1500
corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
gt1 = np.zeros((400, 784))
gt2 = np.zeros((600, 400))
gt3 = np.zeros((10, 600))
# ots = np.zeros(obj, dtype=int)
# dts = np.zeros((obj,5,784))
count = 0
spars = np.zeros(TEST_SAMPLES)
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
        for i in range(TEST_SAMPLES):
            # print(i)
            outputs[i] = net.forward(data, sample=True)
            # print(output)
        outputs[TEST_SAMPLES] = net(data, sample=True)
        outputs[TEST_SAMPLES + 1] = net(data, sample=False)
        output = outputs.mean(0)
        preds = outputs.max(2, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1]  # index of max log-probability
        corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
        correct += pred.eq(target.view_as(pred)).sum().item()
        mask = pred.eq(target.view_as(pred)).eq(0).cpu().numpy()
        # print(mask)
        r = np.where(mask == 1)[1]
        # print(r)
        # r = torch.index_select(pred, mask)
        # l=r#r.cpu().numpy()-1
        if (len(r) > 0):
            # print(count)
            if (count == 0):
                ots = target[r]
                dts = data[r]
                pts = pred[r]
            else:
                ots = torch.cat((ots, target[r]))
                pts = torch.cat((pts, pred[r]))
                dts = torch.cat((dts, data[r]))
            count += len(r)
        if (count >= obj):
            break

for index, num in enumerate(corrects):
    if index < TEST_SAMPLES:
        print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
    elif index == TEST_SAMPLES:
        print('Posterior Mode Accuracy: {}/{}'.format(num, TEST_SIZE))
    elif index == TEST_SAMPLES + 1:
        print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))

print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
corrects = np.append(corrects, correct)

# %%

# plot some difficult to handle in domain data
sns.set_style("dark")
# show(make_grid(dts.cpu()))
net.eval()
res = net.forward(dts, sample=True)
fmnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
fmnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(dts, sample=True)
    fmnist_outputs = np.append(fmnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    fmnist_means = fmnist_means + sigmoid(res.detach().cpu().numpy())
fmnist_means /= 100
sns.set_style("dark")
fig, ax = plt.subplots(30, 2, figsize=(10, 30), gridspec_kw={'width_ratios': [1, 8]})
j = -1
for i in range(30, obj, 3):
    if (pts[i] == ots[i]):
        continue
    else:
        j += 1

    print(j)

    ax[j, 1].set_ylim(0, 100)
    if (j == 29):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10), ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    else:
        ax[j, 1].set_xticks([], [])
    ax[j, 1].set_ylabel("Count")
    ax[j, 1].set_yticks(range(50, 101, 50))
    n, bins, patches = ax[j, 1].hist(fmnist_outputs[i], np.arange(-0.5, 10, 1), color="b")
    patches[pts[i]].set_fc('r')
    patches[ots[i]].set_fc('g')
    ax[j, 1].plot(range(10), fmnist_means[i] / np.sum(fmnist_means[i]) * 100, lw=2, color="y")
    ax[j, 1].axhline(95, color='k', linestyle='dashed', linewidth=1)
    ax[j, 0].imshow(dts.cpu().numpy()[i, 0, 0:27, 0:27])
    ax[j, 0].set_xticks([], [])
    ax[j, 0].set_yticks([], [])
    ax[j, 0].set_ylabel("Truth")
    if (j == 29):
        break
plt.sca(ax[29, 1])
plt.xticks(range(10),
           ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
plt.show()

# %%

# check out-of-domain (MNIST) hard-classification accuacies
net.eval()
correct = 0
corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
gt1 = np.zeros((400, 784))
gt2 = np.zeros((600, 400))
gt3 = np.zeros((10, 600))
# ots = np.zeros(obj, dtype=int)
# dts = np.zeros((obj,5,784))
entropies = np.zeros(10)
count = 0
k = 0
spars = np.zeros(TEST_SAMPLES)
with torch.no_grad():
    for data, target in mnist_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
        for i in range(TEST_SAMPLES):
            # print(i)
            outputs[i] = net.forward(data, sample=True)
            if (i == 0):
                fmnist_means = sigmoid(outputs[i].detach().cpu().numpy())
                for j in range(TEST_BATCH_SIZE):
                    fmnist_means[j] /= np.sum(fmnist_means[j])
            else:
                tmp = sigmoid(outputs[i].detach().cpu().numpy())
                for j in range(TEST_BATCH_SIZE):
                    tmp[j] /= np.sum(tmp[j])
                # print(sum(tmp[j]))
                fmnist_means = fmnist_means + tmp
        fmnist_means /= TEST_SAMPLES
        # print(np.sum(fmnist_means))
        for j in range(TEST_BATCH_SIZE):
            if k == 0 and j == 0:
                entropies = -np.sum(fmnist_means[j] * np.log(fmnist_means[j]))
            else:
                entropies = np.append(entropies, -np.sum(fmnist_means[j] * np.log(fmnist_means[j])))
        k += 1
        outputs[TEST_SAMPLES] = net(data, sample=True)
        outputs[TEST_SAMPLES + 1] = net(data, sample=False)
        output = outputs.mean(0)
        preds = outputs.max(2, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1]  # index of max log-probability
        corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
        correct += pred.eq(target.view_as(pred)).sum().item()

for index, num in enumerate(corrects):
    if index < TEST_SAMPLES:
        print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
    elif index == TEST_SAMPLES:
        print('Posterior Mode Accuracy: {}/{}'.format(num, TEST_SIZE))
    elif index == TEST_SAMPLES + 1:
        print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))

print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
corrects = np.append(corrects, correct)

# %%


