
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
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(1)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 5
COND_OPT = False
CLASSES = 10
SAMPLES = 1
TEST_SAMPLES = 10
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 250
pepochs = 50

# define the data loaders
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
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


# define Gaussian distribution
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def rsample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

    def log_prob_iid(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2))

    def full_log_prob(self, input, gamma):
        return (torch.log(gamma * (torch.exp(self.log_prob_iid(input)))
                          + (1 - gamma) + 1e-8)).sum()


# define Bernoulli distribution
class Bernoulli(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.exact = False

    def rsample(self):
        if self.exact:
            gamma = torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)
        else:
            gamma = torch.distributions.RelaxedBernoulli(probs=self.alpha, temperature=TEMPER_PRIOR).rsample()
        # Variable((self.bern.sample(alpha.size()).to(DEVICE)<alpha.type(torch.cuda.FloatTensor)).type(torch.cuda.FloatTensor)).to(DEVICE)
        return gamma

    def sample(self):
        self.bern.sample()

    def log_prob(self, input):
        if self.exact:
            gamma = torch.round(input.detach())
            output = (gamma * torch.log(self.alpha + 1e-8) + (1 - gamma) * torch.log(1 - self.alpha + 1e-8)).sum()
        else:
            output = (input * torch.log(self.alpha + 1e-8) + (1 - input) * torch.log(1 - self.alpha + 1e-8)).sum()
        return output


# define Normal-Gamma distribution
class GaussGamma(object):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.exact = False
        self.sigma = torch.distributions.Gamma(self.a, self.b)

    def log_prob(self, input, gamma):
        tau = self.sigma.rsample()
        if self.exact:
            gamma1 = torch.round(gamma.detach())
            output = (gamma1 * (self.a * torch.log(self.b) + (self.a - 0.5) * tau - self.b * tau - torch.lgamma(
                self.a) - 0.5 * torch.log(torch.tensor(2 * np.pi))) - tau * torch.pow(input, 2) + (
                                  1 - gamma1) + 1e-8).sum()
        else:
            output = (gamma * (self.a * torch.log(self.b) + (self.a - 0.5) * tau - self.b * tau - torch.lgamma(
                self.a) - 0.5 * torch.log(torch.tensor(2 * np.pi))) - tau * torch.pow(input, 2) + (
                                  1 - gamma) + 1e-8).sum()
        return output


# define BetaBinomial distibution
class BetaBinomial(object):
    def __init__(self, pa, pb):
        super().__init__()
        self.pa = pa
        self.pb = pb
        self.exact = False

    def log_prob(self, input, pa, pb):
        if self.exact:
            gamma = torch.round(input.detach())
        else:
            gamma = input
        return (torch.lgamma(torch.ones_like(input)) + torch.lgamma(gamma + torch.ones_like(input) * self.pa)
                + torch.lgamma(torch.ones_like(input) * (1 + self.pb) - gamma) + torch.lgamma(
                    torch.ones_like(input) * (self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa + gamma)
                - torch.lgamma(torch.ones_like(input) * 2 - gamma) - torch.lgamma(
                    torch.ones_like(input) * (1 + self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa) - torch.lgamma(torch.ones_like(input) * self.pb)).sum()

    def rsample(self):
        gamma = torch.distributions.RelaxedBernoulli(
            probs=torch.distributions.Beta(self.pa, self.pb).rsample().to(DEVICE), temperature=0.001).rsample().to(
            DEVICE)
        return gamma


# define the linear layer for the BNN
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, layer_id):
        super().__init__()

        # configuration of the layer
        self.layer = layer_id
        self.in_features = in_features
        self.out_features = out_features

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # weight priors
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(0, 1))
        self.gammas = torch.Tensor(out_features, in_features).uniform_(0.99, 1)
        self.alpha = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha)
        # model priors
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1, 1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)

        # bias (intercept) parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # bias (intercept) priors
        self.bias_a = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_b = nn.Parameter(torch.Tensor(out_features).uniform_(1, 1.1))
        self.bias_prior = GaussGamma(self.bias_a, self.bias_b)

        # scalars
        self.log_prior = 0
        self.log_variational_posterior = 0
        self.lagrangian = 0

    # forward path
    def forward(self, input, cgamma, sample=False, medimean=False, calculate_log_probs=False):
        # if sampling
        if self.training or sample:
            self.gammas = cgamma
            weight = cgamma * self.weight.rsample()
            bias = self.bias.rsample()
        # if mean of the given model (e.g.) median probability model
        elif medimean:
            weight = cgamma * self.weight.mu
            bias = self.bias.mu
        # if joint mean in the space of models and parameters (for a given alpha vector)
        else:
            weight = self.alpha * self.weight.mu
            bias = self.bias.mu
        # calculate the losses
        if self.training or calculate_log_probs:
            self.alpha = 1 / (1 + torch.exp(-self.lambdal))
            self.log_prior = self.weight_prior.log_prob(weight, cgamma) + self.bias_prior.log_prob(bias,
                                                                                                   torch.ones_like(
                                                                                                       bias)) + self.gamma_prior.log_prob(
                cgamma, pa=self.pa, pb=self.pb)
            self.log_variational_posterior = self.weight.full_log_prob(input=weight,
                                                                       gamma=cgamma) + self.gamma.log_prob(
                cgamma) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior
        # propogate
        return F.linear(input, weight, bias)

    # deine the whole BNN


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(28 * 28, 400, 1)
        self.l2 = BayesianLinear(400, 600, 1)
        self.l3 = BayesianLinear(600, 10, 1)

    def forward(self, x, g1, g2, g3, sample=False, medimean=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1.forward(x, g1, sample, medimean))
        x = F.relu(self.l2.forward(x, g2, sample, medimean))
        x = F.log_softmax(F.relu(self.l3.forward(x, g3, sample, medimean)), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

    # sample the marginal likelihood lower bound
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        negative_log_likelihoods = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            # get the inclusion probabilities for all layers
            self.l1.alpha = 1 / (1 + torch.exp(-self.l1.lambdal))
            self.l1.gamma.alpha = self.l1.alpha
            self.l2.alpha = 1 / (1 + torch.exp(-self.l2.lambdal))
            self.l2.gamma.alpha = self.l2.alpha
            self.l3.alpha = 1 / (1 + torch.exp(-self.l3.lambdal))
            self.l3.gamma.alpha = self.l3.alpha

            # sample the model
            cgamma1 = self.l1.gamma.rsample().to(DEVICE)
            cgamma2 = self.l2.gamma.rsample().to(DEVICE)
            cgamma3 = self.l3.gamma.rsample().to(DEVICE)

            # get the results
            outputs[i] = self.forward(input, g1=cgamma1, g2=cgamma2, g3=cgamma3, sample=True, medimean=False)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            negative_log_likelihoods[i] = F.nll_loss(outputs[i], target, reduction='sum')

        # the current log prior
        log_prior = log_priors.mean()
        # the current log variational posterior
        log_variational_posterior = log_variational_posteriors.mean()
        # the current negative log likelihood
        negative_log_likelihood = negative_log_likelihoods.mean()

        # the current ELBO
        loss = negative_log_likelihood + (log_variational_posterior - log_prior) / NUM_BATCHES
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


# save the relevant parameters histograms
def write_weight_histograms(epoch, i):
    writer.add_histogram('histogram/mfw1_mu', net.l1.weight_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfw1_rho', net.l1.weight_rho,epoch+i*epochs)
    writer.add_histogram('histogram/mfw2_mu', net.l2.weight_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfw2_rho', net.l2.weight_rho,epoch+i*epochs)
    writer.add_histogram('histogram/mfw3_mu', net.l3.weight_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfw3_rho', net.l3.weight_rho,epoch+i*epochs)
    writer.add_histogram('histogram/mfb1_mu', net.l1.bias_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfb1_rho', net.l1.bias_rho,epoch+i*epochs)
    writer.add_histogram('histogram/mfb2_mu', net.l2.bias_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfb2_rho', net.l2.bias_rho,epoch+i*epochs)
    writer.add_histogram('histogram/mfb3_mu', net.l3.bias_mu, epoch + i * epochs)
    # writer.add_histogram('histogram/mfb3_rho', net.l3.bias_rho,epoch+i*epochs)


# save the relevant losses
def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    writer.add_scalar('logs/mloss', loss, epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar('logs/mcomplexity_cost', log_variational_posterior - log_prior,
                      i * epochs * NUM_BATCHES + epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar('logs/mlog_prior', log_prior, i * epochs * NUM_BATCHES + epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar('logs/mlog_variational_posterior', log_variational_posterior,
                      i * epochs * NUM_BATCHES + epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar('logs/mnegative_log_likelihood', negative_log_likelihood,
                      i * epochs * NUM_BATCHES + epoch * NUM_BATCHES + batch_idx)


# Stochastic Variational Inference iteration
def train(net, optimizer, epoch, i):
    net.train()
    if epoch == 0:  # write initial distributions
        write_weight_histograms(epoch, i)

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


def test_ensemble(net):
    net.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    cases3 = 0
    cases4 = 0
    ctr = 0
    corrects = np.zeros(TEST_SAMPLES + 12, dtype=int)
    spars = np.zeros(TEST_SAMPLES)
    gt1 = np.zeros((400, 784))
    gt2 = np.zeros((600, 400))
    gt3 = np.zeros((10, 600))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES + 12, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                # get the inclusion probabilities for all layers
                net.l1.alpha = 1 / (1 + torch.exp(-net.l1.lambdal))
                net.l1.gamma.alpha = net.l1.alpha
                net.l2.alpha = 1 / (1 + torch.exp(-net.l2.lambdal))
                net.l2.gamma.alpha = net.l2.alpha
                net.l3.alpha = 1 / (1 + torch.exp(-net.l3.lambdal))
                net.l3.gamma.alpha = net.l3.alpha

                # sample the model
                g1 = net.l1.gamma.rsample().to(DEVICE)
                g2 = net.l2.gamma.rsample().to(DEVICE)
                g3 = net.l3.gamma.rsample().to(DEVICE)
                ctr += 1
                spars[i] = spars[i] + ((torch.sum(g1 > 0.5).cpu().detach().numpy() + torch.sum(
                    g2 > 0.5).cpu().detach().numpy() + torch.sum(g3 > 0.5).cpu().detach().numpy()) / (
                                                   10 * 600 + 400 * 600 + 400 * 784))
                gt1 = gt1 + (g1 > 0.5).cpu().numpy()
                gt2 = gt2 + (g2 > 0.5).cpu().numpy()
                gt3 = gt3 + (g3 > 0.5).cpu().numpy()
                outputs[i] = net.forward(data, sample=True, medimean=False, g1=net.l1.gamma.rsample(),
                                         g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample())
                outputs[i + 10] = net.forward(data, sample=True, medimean=False,
                                              g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                              g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                              g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
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
                if (i == 0):
                    mydata_means_med = sigmoid(outputs[i + 10].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means_med[j] /= np.sum(mydata_means_med[j])
                else:
                    tmp = sigmoid(outputs[i + 10].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means_med = mydata_means_med + tmp
            mydata_means /= TEST_SAMPLES
            mydata_means_med /= TEST_SAMPLES

            outputs[TEST_SAMPLES + 11] = net(data, sample=False, medimean=True,
                                             g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
            outputs[TEST_SAMPLES + 10] = net(data, sample=False, medimean=False, g1=net.l1.gamma.rsample(),
                                             g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample())

            output1 = outputs[0:10].mean(0)
            output2 = outputs[10:20].mean(0)

            preds = outputs.max(2, keepdim=True)[1]
            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
            pred2 = output2.max(1, keepdim=True)[1]

            corrects += preds.eq(target.view_as(pred1)).sum(dim=1).squeeze().cpu().numpy()
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()

            # print(mydata_means[1][1])
            for jj in range(TEST_BATCH_SIZE):
                if mydata_means[jj][pred1.detach().cpu().numpy()[jj]] >= 0.95:
                    correct3 += pred1[jj].eq(target.view_as(pred1)[jj]).sum().item()
                    cases3 += 1

                if mydata_means_med[jj][pred2.detach().cpu().numpy()[jj]] >= 0.95:
                    correct4 += pred2[jj].eq(target.view_as(pred2)[jj]).sum().item()
                    cases4 += 1

    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        elif index < TEST_SAMPLES + 10:
            print('Component MPM {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        elif index == TEST_SAMPLES + 10:
            print('Posterior Mode Accuracy: {}/{}'.format(num, TEST_SIZE))
        elif index == TEST_SAMPLES + 11:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
        elif index == TEST_SAMPLES + 12:
            print('Posterior MPM Mean Accuracy: {}/{}'.format(num, TEST_SIZE))

    print('Ensemble Accuracy: {}/{}'.format(correct1, TEST_SIZE))
    print('Median Ensemble Accuracy: {}/{}'.format(correct2, TEST_SIZE))

    corrects = np.append(corrects, correct1)
    corrects = np.append(corrects, correct2)

    corrects = np.append(corrects, correct3 / cases3)
    corrects = np.append(corrects, cases3)
    corrects = np.append(corrects, correct4 / cases4)
    corrects = np.append(corrects, cases4)

    ps = ((np.sum(gt1 > 0) + np.sum(gt2 > 0) + np.sum(gt3 > 0)) / (10 * 600 + 400 * 600 + 400 * 784)) / 10
    print(spars / ctr)

    corrects = np.append(corrects, ps)
    corrects = np.append(corrects, np.median(spars) / ctr)
    return corrects


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def outofsample(net, medimod = False):
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
                if medimod:
                    outputs[i] = net.forward(data, sample=True, medimean=False,
                                             g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
                else:
                    outputs[i] = net.forward(data, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                             g3=net.l3.gamma.rsample(), sample=True)
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
            # print(np.sum(mydata_means))
            for j in range(TEST_BATCH_SIZE):
                if k == 0 and j == 0:
                    entropies = -np.sum(mydata_means[j] * np.log(mydata_means[j]))
                else:
                    entropies = np.append(entropies, -np.sum(mydata_means[j] * np.log(mydata_means[j])))
            k += 1
            output = outputs[1:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    cdf(entropies.flatten())
    return entropies.flatten()


def ptrain(net, optimizer, epoch, i):
    net.train()
    if epoch == 0:  # write initial distributions
        write_weight_histograms(epoch, i)
        net.l1.lambdal.requires_grad = False
        net.l2.lambdal.requires_grad = False
        net.l3.lambdal.requires_grad = False
        net.l1.pa.requires_grad = False
        net.l2.pa.requires_grad = False
        net.l3.pa.requires_grad = False
        net.l1.pb.requires_grad = False
        net.l2.pb.requires_grad = False
        net.l3.pb.requires_grad = False
        net.l1.weight_a.requires_grad = False
        net.l2.weight_a.requires_grad = False
        net.l3.weight_a.requires_grad = False
        net.l1.weight_b.requires_grad = False
        net.l2.weight_b.requires_grad = False
        net.l3.weight_b.requires_grad = False
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


print("Classes loaded")

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
        {'params': net.l1.pa, 'lr': 0.001},
        {'params': net.l2.pa, 'lr': 0.001},
        {'params': net.l3.pa, 'lr': 0.001},
        {'params': net.l1.pb, 'lr': 0.001},
        {'params': net.l2.pb, 'lr': 0.001},
        {'params': net.l3.pb, 'lr': 0.001},
        {'params': net.l1.weight_a, 'lr': 0.00001},
        {'params': net.l2.weight_a, 'lr': 0.00001},
        {'params': net.l3.weight_a, 'lr': 0.00001},
        {'params': net.l1.weight_b, 'lr': 0.00001},
        {'params': net.l2.weight_b, 'lr': 0.00001},
        {'params': net.l3.weight_b, 'lr': 0.00001},
        {'params': net.l1.bias_a, 'lr': 0.00001},
        {'params': net.l2.bias_a, 'lr': 0.00001},
        {'params': net.l3.bias_a, 'lr': 0.00001},
        {'params': net.l1.bias_b, 'lr': 0.00001},
        {'params': net.l2.bias_b, 'lr': 0.00001},
        {'params': net.l3.bias_b, 'lr': 0.00001},
        {'params': net.l1.lambdal, 'lr': 0.1},
        {'params': net.l2.lambdal, 'lr': 0.1},
        {'params': net.l3.lambdal, 'lr': 0.1}
    ], lr=0.0001)
    for epoch in range(epochs):
        if (net.l1.pa / (net.l1.pa + net.l1.pb)).mean() < 0.1 or epoch == 20:
            print(epoch)
            net.l1.gamma_prior.exact = True
            net.l2.gamma_prior.exact = True
            net.l3.gamma_prior.exact = True
            net.l1.bias_prior.exact = True
            net.l2.bias_prior.exact = True
            net.l3.bias_prior.exact = True
            net.l1.weight_prior.exact = True
            net.l1.weight_prior.exact = True
            net.l1.weight_prior.exact = True
            # res = test_ensemble(net)
            # res = np.append(res,net.l1.alpha.cpu().detach().numpy().mean())
            # res = np.append(res,net.l2.alpha.cpu().detach().numpy().mean())
            # res = np.append(res,net.l3.alpha.cpu().detach().numpy().mean())
            # np.savetxt("fmaccuracieshalf_"+str(i)+".csv", res, delimiter=",")
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
                {'params': net.l1.pa, 'lr': 0.00},
                {'params': net.l2.pa, 'lr': 0.00},
                {'params': net.l3.pa, 'lr': 0.00},
                {'params': net.l1.pb, 'lr': 0.00},
                {'params': net.l2.pb, 'lr': 0.00},
                {'params': net.l3.pb, 'lr': 0.00},
                {'params': net.l1.weight_a, 'lr': 0.00},
                {'params': net.l2.weight_a, 'lr': 0.00},
                {'params': net.l3.weight_a, 'lr': 0.00},
                {'params': net.l1.weight_b, 'lr': 0.00},
                {'params': net.l2.weight_b, 'lr': 0.00},
                {'params': net.l3.weight_b, 'lr': 0.00},
                {'params': net.l1.bias_a, 'lr': 0.00},
                {'params': net.l2.bias_a, 'lr': 0.00},
                {'params': net.l3.bias_a, 'lr': 0.00},
                {'params': net.l1.bias_b, 'lr': 0.00},
                {'params': net.l2.bias_b, 'lr': 0.00},
                {'params': net.l3.bias_b, 'lr': 0.00},
                {'params': net.l1.lambdal, 'lr': 0.0001},
                {'params': net.l2.lambdal, 'lr': 0.0001},
                {'params': net.l3.lambdal, 'lr': 0.0001}
            ], lr=0.0001)
        train(net, optimizer, epoch, i)
        print(net.l1.alpha.cpu().detach().numpy().mean())
        print(net.l2.alpha.cpu().detach().numpy().mean())
        print(net.l3.alpha.cpu().detach().numpy().mean())
        print((net.l1.weight_a).mean())
        print((net.l1.weight_b).mean())
        print((net.l1.pa / (net.l1.pa + net.l1.pb)).mean())
        print((net.l2.pa / (net.l2.pa + net.l2.pb)).mean())
        print((net.l3.pa / (net.l3.pa + net.l3.pb)).mean())
    net.l1.alpha.data = 1 / (1 + torch.exp(-net.l1.lambdal.data))  # (torch.clamp(self.l1.alpha.data,1e-8 , 1-1e-8))
    net.l2.alpha.data = 1 / (1 + torch.exp(-net.l2.lambdal.data))  # (torch.clamp(self.l2.alpha.data,1e-8 , 1-1e-8))
    net.l3.alpha.data = 1 / (1 + torch.exp(-net.l3.lambdal.data))  # (torch.clamp(self.l3.alpha.data,1e-8 , 1-1e-8))
    net.l1.gamma.alpha.data = 1 / (
                1 + torch.exp(-net.l1.lambdal.data))  # (torch.clamp(self.l1.alpha.data,1e-8 , 1-1e-8))
    net.l2.gamma.alpha.data = 1 / (
                1 + torch.exp(-net.l2.lambdal.data))  # (torch.clamp(self.l2.alpha.data,1e-8 , 1-1e-8))
    net.l3.gamma.alpha.data = 1 / (
                1 + torch.exp(-net.l3.lambdal.data))  # (torch.clamp(self.l3.alpha.data,1e-8 , 1-1e-8))

    net.l1.gamma.exact = True
    net.l2.gamma.exact = True
    net.l3.gamma.exact = True

    res = test_ensemble(net)

    os = (torch.sum(net.l1.alpha.data > 0.5).cpu().detach().numpy() + torch.sum(
        net.l2.alpha.data > 0.5).cpu().detach().numpy() + torch.sum(net.l3.alpha.data > 0.5).cpu().detach().numpy()) / (
                     10 * 600 + 400 * 600 + 400 * 784)

    res = np.append(res, os)

    res = np.append(res, net.l1.alpha.cpu().detach().numpy().mean())
    res = np.append(res, net.l2.alpha.cpu().detach().numpy().mean())
    res = np.append(res, net.l3.alpha.cpu().detach().numpy().mean())

    np.savetxt("hypermaccuraciesffm_" + str(i) + ".csv", res, delimiter=",")

    print(net.l1.alpha.cpu().detach().numpy().mean())
    print(net.l2.alpha.cpu().detach().numpy().mean())
    print(net.l3.alpha.cpu().detach().numpy().mean())

    plt.hist(net.l3.gamma.alpha.data.view(-1).cpu().detach().numpy(), bins=1000)
    plt.show()
    plt.close()
    plt.hist(net.l1.alpha.view(-1).cpu().detach().numpy(), bins=100, color="skyblue", ec="skyblue")
    plt.show()
    plt.savefig('./results/hyperhfmn1fg' + str(i) + '.png')
    plt.close()
    plt.hist(net.l2.alpha.view(-1).cpu().detach().numpy(), bins=100, color="skyblue", ec="skyblue")
    plt.show()
    plt.savefig('./results/hyperhfmn2fg' + str(i) + '.png')
    plt.close()
    plt.hist(net.l3.alpha.view(-1).cpu().detach().numpy(), bins=100, color="skyblue", ec="skyblue")
    plt.show()
    plt.savefig('./results/hyperhfmn3fg' + str(i) + '.png')
    plt.close()

    enr = outofsample(net)
    np.savetxt(X=enr, fname="hyperentrofullffmmfm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "hyperffmfm" + str(i) + ".par")
    enr = outofsample(net, True)
    np.savetxt(X=enr, fname="mpmhyperentrofullffmmfm" + str(i) + ".csv", delimiter=",")
    # pepochs = 1
    for epoch in range(pepochs):
        ptrain(net, optimizer, epoch, i)
    res = test_ensemble(net)
    np.savetxt("hyperptmmaccuraciesffm_" + str(i) + ".csv", res, delimiter=",")
    enr = outofsample(net)
    np.savetxt(X=enr, fname="hyperpentrofullmmfffm" + str(i) + ".csv", delimiter=",")
    enr = outofsample(net, True)
    np.savetxt(X=enr, fname="mpmhyperpentrofullmmfffm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "hyperffmpfm" + str(i) + ".par")

# %%

i = 9

#uncomment to restore the model
#net = BayesianNetwork().to(DEVICE)
#net.load_state_dict(torch.load("hyperffmpfm" + str(i) + ".par"))
#net.eval()
#net.l1.alpha.data = 1 / (1 + torch.exp(-net.l1.lambdal.data))  # (torch.clamp(self.l1.alpha.data,1e-8 , 1-1e-8))
#net.l2.alpha.data = 1 / (1 + torch.exp(-net.l2.lambdal.data))  # (torch.clamp(self.l2.alpha.data,1e-8 , 1-1e-8))
#net.l3.alpha.data = 1 / (1 + torch.exp(-net.l3.lambdal.data))


# %%
#see parameters of the trained models
print(net.l1.alpha.cpu().detach().numpy().mean())
print(net.l2.alpha.cpu().detach().numpy().mean())
print(net.l3.alpha.cpu().detach().numpy().mean())
print(i)


net.eval()
correct = 0
corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
spars = np.zeros(TEST_SAMPLES)
gt1 = np.zeros((400, 784))
gt2 = np.zeros((600, 400))
gt3 = np.zeros((10, 600))
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
        for i in range(TEST_SAMPLES):
            # print(i)
            g1 = net.l1.gamma.rsample()
            g2 = net.l2.gamma.rsample()
            g3 = net.l3.gamma.rsample()
            spars[i] = spars[i] + ((torch.sum(g1 > 0.5).cpu().detach().numpy() + torch.sum(
                g2 > 0.5).cpu().detach().numpy() + torch.sum(g3 > 0.5).cpu().detach().numpy()) / (
                                               10 * 600 + 400 * 600 + 400 * 784))

            gt1 = gt1 + (g1 > 0.5).cpu().numpy()
            gt2 = gt2 + (g2 > 0.5).cpu().numpy()
            gt3 = gt3 + (g3 > 0.5).cpu().numpy()
            outputs[i] = net.forward(data, sample=True, g1=g1, g2=g2, g3=g3)
            # print(output)
        outputs[TEST_SAMPLES] = net(data, sample=True,
                                    g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                    g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                    g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
        outputs[TEST_SAMPLES + 1] = net(data, sample=False, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                        g3=net.l3.gamma.rsample())
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

print(pred)
print(target.view_as(pred))
corrects
preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()

# %%

# sparcities from the individual samples
print(spars / 10 / 200)

# %%

# posterior model averaging sparsity
((np.sum(gt1 > 0) + np.sum(gt2 > 0) + np.sum(gt3 > 0)) / (10 * 600 + 400 * 600 + 400 * 784)) / 10

# %%


# histograms of the marginal inclusion probabilities
plt.hist(net.l1.alpha.view(-1).cpu().detach().numpy(), bins=20)
plt.show()
plt.savefig('./results/hyperhfmn1fgn' + str(i) + '.png')
plt.close()
plt.hist(net.l2.alpha.view(-1).cpu().detach().numpy(), bins=20)
plt.show()
plt.savefig('./results/hyperhfmn2fgn' + str(i) + '.png')
plt.close()
plt.hist(net.l3.alpha.view(-1).cpu().detach().numpy(), bins=20)
plt.show()
plt.savefig('./results/hyperhfmn3fgn' + str(i) + '.png')
plt.close()

# %%

# show all marginal inclusion probabilities
plt.imshow(net.l1.gamma.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/miphyperfmn1fg' + str(i) + '.png')
plt.close()
plt.imshow(net.l2.gamma.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/miphyperfmn2fg' + str(i) + '.png')
plt.close()
plt.imshow(net.l3.gamma.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/miphyperfmn3fg' + str(i) + '.png')
plt.close()

#show samples from the weight parameters
plt.imshow(net.l1.weight.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/wephyperfmn1fg' + str(i) + '.png')
plt.close()
plt.imshow(net.l2.weight.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/wephyperfmn2fg' + str(i) + '.png')
plt.close()
plt.imshow(net.l3.weight.rsample().cpu().detach().numpy())
plt.show()
plt.savefig('./results/wephyperfmn3fg' + str(i) + '.png')
plt.close()


# %%
#show the correlation structures between the latent states
latents = net.l1.gamma.rsample()
gammas = net.l1.weight.rsample().cpu().detach().numpy()
latents = latents.cpu().detach().numpy()

cormat = np.abs(np.corrcoef(latents))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/latcorl1mfmn' + str(i) + '.png')
plt.close()

print(np.mean(cormat))
cormat = np.abs(np.corrcoef(gammas))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/gamcorl1mfmn' + str(i) + '.png')
plt.close()
print(np.mean(cormat))

latents = net.l2.gamma.rsample()
gammas = net.l2.weight.rsample().cpu().detach().numpy()
latents = latents.cpu().detach().numpy()

cormat = np.abs(np.corrcoef(latents))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/latcorl2mfmn' + str(i) + '.png')
plt.close()
print(np.mean(cormat))
cormat = np.abs(np.corrcoef(gammas))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/gamcorl2mfmn' + str(i) + '.png')
plt.close()
print(np.mean(cormat))

latents = net.l3.gamma.rsample()
gammas = net.l3.weight.rsample().cpu().detach().numpy()
latents = latents.cpu().detach().numpy()

cormat = np.abs(np.corrcoef(latents))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/latcorl3mfmn' + str(i) + '.png')
plt.close()
print(np.mean(cormat))
cormat = np.abs(np.corrcoef(gammas))
plt.imshow(cormat)
plt.show()
plt.savefig('./results/gamcorl3mfmn' + str(i) + '.png')
plt.close()
print(np.mean(cormat))

# %%

# show all weights

# %%

SIM_TEST_SAMPLES = 1
# torch.cuda.empty_cache()
MAX_SAMPLES =10
net.eval()
correct = 0
corrects = np.zeros(MAX_SAMPLES, dtype=int)
with torch.no_grad():
    for SIM_TEST_SAMPLES in range(MAX_SAMPLES-2,MAX_SAMPLES):
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(SIM_TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(SIM_TEST_SAMPLES + 1):
                # if SIM_TEST_SAMPLES>0 and i< SIM_TEST_SAMPLES:
                #    outputs[i]=tmp[i]
                # else:
                outputs[i] = net.forward(data, sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                         g3=net.l3.gamma.rsample())
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
                g1 = net.l1.gamma.rsample()
                g2 = net.l2.gamma.rsample()
                g3 = net.l3.gamma.rsample()
                gt1 = gt1 + (g1 > 0.5).cpu().numpy()
                gt2 = gt2 + (g2 > 0.5).cpu().numpy()
                gt3 = gt3 + (g3 > 0.5).cpu().numpy()
                outputs[i] = net.forward(data, sample=True, g1=g1, g2=g2, g3=g3)
                # print(output)
            outputs[TEST_SAMPLES] = net(data, sample=True,
                                        g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                        g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                        g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
            outputs[TEST_SAMPLES + 1] = net(data, sample=False, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                            g3=net.l3.gamma.rsample())
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
                print(count)
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

def show(img,outg):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    plt.savefig(outg + str(i) + '.png')
    plt.close()


# %% show insample/ out of sample results

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


fmnist_sample = iter(test_loader).next()
fmnist_sample[0] = fmnist_sample[0].to(DEVICE)
print(fmnist_sample[1])
sns.set_style("dark")
show(make_grid(fmnist_sample[0].cpu()),'./results/samplefmnist')

net.eval()
res = net.forward(fmnist_sample[0], sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                  g3=net.l3.gamma.rsample())
fmnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
fmnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(fmnist_sample[0], sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                      g3=net.l3.gamma.rsample())
    fmnist_outputs = np.append(fmnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    fmnist_means = fmnist_means + sigmoid(res.detach().cpu().numpy())
fmnist_means /= 100
# fmnist_means = fmnist_means/np.sum(np.reshape(fmnist_means,(5,10)),axis=0)
# fmnist_means = sigmoid(fmnist_means)
sns.set_style("dark")
fig, ax = plt.subplots(5, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 9]})
for j in range(5):
    if (j == 4):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10),
                            ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle Boot"])
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
plt.savefig('./results/fmnistindomain' + str(i) + '.png')
plt.close()


# %%

mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=5, shuffle=False)
mnist_sample = iter(mnist_loader).next()
mnist_sample[0] = mnist_sample[0].to(DEVICE)
print(mnist_sample[1])
sns.set_style("dark")
show(make_grid(mnist_sample[0].cpu()),'./results/samplemnist')

net.eval()
res = net.forward(mnist_sample[0], sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                  g3=net.l3.gamma.rsample())
mnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
mnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(mnist_sample[0], sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                      g3=net.l3.gamma.rsample())
    mnist_outputs = np.append(mnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    mnist_means = mnist_means + sigmoid(res.detach().cpu().numpy())
mnist_means /= 100
sns.set_style("dark")
fig, ax = plt.subplots(5, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 9]})
for j in range(5):
    if (j == 4):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10),
                            ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle Boot"])
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
plt.xticks(range(10),
           ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
plt.show()
plt.savefig('./results/fmnistoutdomain' + str(i) + '.png')
plt.close()
# %%


#show difficult cases
sns.set_style("dark")
# show(make_grid(dts.cpu()))
net.eval()
res = net.forward(dts, sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample())
fmnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
fmnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res = net.forward(dts, sample=True, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(), g3=net.l3.gamma.rsample())
    fmnist_outputs = np.append(fmnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    fmnist_means = fmnist_means + sigmoid(res.detach().cpu().numpy())
fmnist_means /= 100
sns.set_style("dark")
fig, ax = plt.subplots(16, 2, figsize=(11,16), gridspec_kw={'width_ratios': [1, 9]})
j = -1
for i in range(0, len(pts), 3):
    if (pts[i] == ots[i]):
        continue
    else:
        j += 1

    print(j)

    ax[j, 1].set_ylim(0, 100)
    if (j == 19):
        ax[j, 1].set_xlabel("Categories")
        ax[j, 1].set_xticks(range(10),
                            ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle Boot"])
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
    if (j == 15):
        break
i = 9
plt.sca(ax[15, 1])
plt.xticks(range(10),
           ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
plt.show()
plt.savefig('./results/fmnistdifficult' + str(i) + '.png')
plt.close()
# %%

net.eval()


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


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
            outputs[i] = net.forward(data, g1=net.l1.gamma.rsample(), g2=net.l2.gamma.rsample(),
                                     g3=net.l3.gamma.rsample(), sample=True)
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
        output = outputs.mean(0)
        preds = outputs.max(2, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1]  # index of max log-probability
        corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
        correct += pred.eq(target.view_as(pred)).sum().item()


# %%

def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


cdf(entropies.flatten())
np.savetxt(X=entropies.flatten(), fname="full.csv", delimiter=",")

# %%
