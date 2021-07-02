# %%

#% matplotlib
#inline
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

# define the summary writer
writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(2)

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
PI = 1.0
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

            preds = preds = outputs.max(2, keepdim=True)[1]
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

# make inference on 10 networks
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
    np.savetxt(X=enr, fname="entrogaussfmm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "gaussffm" + str(i) + ".par")

    np.savetxt("gaussfmaccuracies_" + str(i) + ".csv", res, delimiter=",")

# %%

# make inference on 10 networks
for i in range(10):
    torch.manual_seed(i)
    print(i)
    net = BayesianNetwork().to(DEVICE)
    net.load_state_dict(torch.load("gaussffm" + str(i) + ".par"))
    print("loaded simple")
    enr = outofsample(net)
    np.savetxt(X=enr, fname="entrogaussfmm" + str(i) + ".csv", delimiter=",")

# %%

epochs = 2


def train(net, optimizer, epoch, i):
    net.train()
    if epoch == 0:  # write initial distributions
        write_weight_histograms(epoch, i)
        # net.l1.alpha.requires_grad = False
        # net.l2.alpha.requires_grad = False
        # net.l3.alpha.requires_grad = False
        net.l1.lambdal.requires_grad = False
        net.l2.lambdal.requires_grad = False
        net.l3.lambdal.requires_grad = False
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()

        if COND_OPT:
            net.l1.weight_mu.grad = net.l1.weight_mu.grad * net.l1.gammas.data
            net.l2.weight_mu.grad = net.l2.weight_mu.grad * net.l2.gammas.data
            net.l3.weight_mu.grad = net.l3.weight_mu.grad * net.l3.gammas.data

        optimizer.step()
        write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood)
    print(epoch + 1)
    print(loss)
    print(negative_log_likelihood)
    print(net.l1.alpha.cpu().detach().numpy().mean())
    print(net.l2.alpha.cpu().detach().numpy().mean())
    print(net.l3.alpha.cpu().detach().numpy().mean())
    write_weight_histograms(epoch + 1, i)


net.zero_grad()
for epoch in range(epochs):
    train(net, optimizer, epoch, i)

# %%

for i in range(10):
    res = test_ensemble(net)
    res = np.append(res, net.l1.alpha.cpu().detach().numpy().mean())
    res = np.append(res, net.l2.alpha.cpu().detach().numpy().mean())
    res = np.append(res, net.l3.alpha.cpu().detach().numpy().mean())
    np.savetxt("ptmaccuracies_" + str(i) + ".csv", res, delimiter=",")

# %%

res

# %%

plt.hist(net.l3.gamma.alpha.data.view(-1).cpu().detach().numpy(), bins=1000)
plt.show()
plt.hist(net.l1.alpha.view(-1).cpu().detach().numpy(), bins=1000)
plt.show()
plt.hist(net.l2.alpha.view(-1).cpu().detach().numpy(), bins=1000)
plt.show()
plt.hist(net.l3.alpha.view(-1).cpu().detach().numpy(), bins=1000)
plt.show()


# %%

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


# %%

fmnist_sample = iter(test_loader).next()
fmnist_sample[0] = fmnist_sample[0].to(DEVICE)
print(fmnist_sample[1])
sns.set_style("dark")
show(make_grid(fmnist_sample[0].cpu()))
net.eval()
fmnist_outputs = net.forward(fmnist_sample[0], sample=True, g1=net.l1.gamma.sample(), g2=net.l2.gamma.sample(),
                             g3=net.l3.gamma.sample()).max(1, keepdim=True)[1].detach().cpu().numpy()
for _ in range(99):
    fmnist_outputs = np.append(fmnist_outputs, net.forward(fmnist_sample[0], sample=True, g1=net.l1.gamma.sample(),
                                                           g2=net.l2.gamma.sample(), g3=net.l3.gamma.sample()).max(1,
                                                                                                                   keepdim=True)[
        1].detach().cpu().numpy(), axis=1)
sns.set_style("darkgrid")
plt.subplots(5, 1, figsize=(10, 4))
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.ylim(0, 100)
    plt.xlabel("Categories")
    plt.xticks(range(10),
               ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
    plt.ylabel("Count")
    plt.yticks(range(50, 101, 50))
    plt.hist(fmnist_outputs[i], np.arange(-0.5, 10, 1))

# %%

mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=5, shuffle=False)
mnist_sample = iter(mnist_loader).next()
mnist_sample[0] = mnist_sample[0].to(DEVICE)
print(mnist_sample[1])
sns.set_style("dark")
show(make_grid(mnist_sample[0].cpu()))

net.eval()
mnist_outputs = \
net(mnist_sample[0], sample=True, g1=net.l1.gamma.sample(), g2=net.l2.gamma.sample(), g3=net.l3.gamma.sample()).max(1,
                                                                                                                    keepdim=True)[
    1].detach().cpu().numpy()
for _ in range(99):
    mnist_outputs = np.append(mnist_outputs,
                              net(mnist_sample[0], sample=True, g1=net.l1.gamma.sample(), g2=net.l2.gamma.sample(),
                                  g3=net.l3.gamma.sample()).max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)

sns.set_style("darkgrid")
plt.subplots(5, 1, figsize=(10, 4))
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.ylim(0, 100)
    plt.xlabel("Categories")
    plt.xticks(range(10),
               ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
    plt.ylabel("Count")
    plt.yticks(range(50, 101, 50))
    plt.hist(mnist_outputs[i], np.arange(-0.5, 10, 1))

# %%

SIM_TEST_SAMPLES = 1
# torch.cuda.empty_cache()
MAX_SAMPLES = 1000
net.eval()
correct = 0
corrects = np.zeros(MAX_SAMPLES, dtype=int)
with torch.no_grad():
    for SIM_TEST_SAMPLES in range(MAX_SAMPLES):
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(SIM_TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(SIM_TEST_SAMPLES + 1):
                # if SIM_TEST_SAMPLES>0 and i< SIM_TEST_SAMPLES:
                #    outputs[i]=tmp[i]
                # else:
                outputs[i] = net.forward(data, sample=True, g1=net.l1.gamma.sample(), g2=net.l2.gamma.sample(),
                                         g3=net.l3.gamma.sample())
            # outputss = outputs[0:(SIM_TEST_SAMPLES+1),0:(TEST_BATCH_SIZE-1),0:(CLASSES-1)]
            output = outputs.mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            corrects[SIM_TEST_SAMPLES] = correct / TEST_SIZE
            # tmp = outputs
        print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
        correct = 0

# %%


