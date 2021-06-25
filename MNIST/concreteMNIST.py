#% matplotlib
#inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import numpy as np

# define the summary writer
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
    datasets.MNIST(
        './mnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
mnist_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fmnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=5,
    shuffle=False)
TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

CLASSES = 10
TRAIN_EPOCHS = 250
SAMPLES = 1
TEST_SAMPLES = 10

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


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
        self.linear1 = nn.Linear(784, 400)
        self.linear2 = nn.Linear(400, 600)
        self.linear3 = nn.Linear(600, 10)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.relu = nn.ReLU()

    def forward(self, x):
        regularization = torch.empty(3, device=x.device)

        x = x.view(-1, 28 * 28)
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
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        y_fit, regularization = net.forward(data)
        loss = concrete_loss(target, y_fit, regularization)
        loss.backward()
        optimizer.step()
    print(epoch + 1)
    print(loss)


def test_ensemble_old():
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

    return corrects


def test_ensemble(net):
    net.eval()
    correct = 0
    correct3 = 0
    cases3 = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # print(target)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i], b = net(data)
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

            outputs[TEST_SAMPLES], b = net(data)
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
                outputs[i], b = net.forward(data)
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
    torch.manual_seed(i)
    wr = 1e-6
    dr = 1e-5
    net = Model(wr, dr).to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train(net, optimizer, epoch, i)

    res = test_ensemble(net)

    res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
                          torch.sigmoid(net.conc_drop3.p_logit)])
    np.savetxt("MnistConDropr_" + str(i) + ".csv", res, delimiter=",")
    enr = outofsample(net)
    np.savetxt(X=enr, fname="entroConcretemmm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "fmmmConcrete" + str(i) + ".par")
    # pepochs = 1
    # net.conc_drop1.parameters.requires_grad = False
    # net.conc_drop2.parameters.requires_grad = False
    # net.conc_drop3.parameters.requires_grad = False
    net.conc_drop1.p_logit.requires_grad = False
    net.conc_drop2.p_logit.requires_grad = False
    net.conc_drop3.p_logit.requires_grad = False
    optimizer = optim.Adam([
        {'params': net.conc_drop1.p_logit, 'lr': 0.0000},
        {'params': net.conc_drop2.p_logit, 'lr': 0.0000},
        {'params': net.conc_drop3.p_logit, 'lr': 0.0000}
    ], lr=0.0001)
    for epoch in range(50):
        train(net, optimizer, epoch, i)
    res = test_ensemble(net)
    res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
                          torch.sigmoid(net.conc_drop3.p_logit)])
    np.savetxt("pMnistConDropr__" + str(i) + ".csv", res, delimiter=",")
    enr = outofsample(net)
    np.savetxt(X=enr, fname="pentroConcretemmm" + str(i) + ".csv", delimiter=",")
    torch.save(net.state_dict(), "fmmmpConcrete" + str(i) + ".par")

# %%

# # make inference on 10 networks
# for i in range(9,10):
#     torch.manual_seed(i)
#     wr = 1e-6
#     dr = 1e-5
#     net = Model(wr, dr).to(DEVICE)
#     net.load_state_dict(torch.load("fmmmConcrete" + str(i) + ".par"))
#     enr = outofsample(net)
#     np.savetxt(X=enr, fname="entroConcretemmm" + str(i) + ".csv", delimiter=",")
#     net.load_state_dict(torch.load("fmmmpConcrete" + str(i) + ".par"))
#     enr = outofsample(net)
#     np.savetxt(X=enr, fname="pentroConcretemmm" + str(i) + ".csv", delimiter=",")
#
#
# # make inference on 10 networks
# for i in range(2, 10):
#     torch.manual_seed(i)
#     wr = 1e-6
#     dr = 1e-5
#     net = Model(wr, dr).to(DEVICE)
#
#     optimizer = optim.Adam(net.parameters(), lr=0.0001)
#     for epoch in range(epochs):
#         train(net, optimizer, epoch, i)
#
#     res = test_ensemble(net)
#
#     res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
#                           torch.sigmoid(net.conc_drop3.p_logit)])
#     np.savetxt("FMnistConDropr_" + str(i) + ".csv", res, delimiter=",")
#     enr = outofsample(net)
#     np.savetxt(X=enr, fname="entroConcretefm" + str(i) + ".csv", delimiter=",")
#     torch.save(net.state_dict(), "ffmConcrete" + str(i) + ".par")
#     # pepochs = 1
#     for epoch in range(pepochs):
#         ptrain(net, optimizer, epoch, i)
#     res = test_ensemble(net)
#     res = np.append(res, [torch.sigmoid(net.conc_drop1.p_logit), torch.sigmoid(net.conc_drop2.p_logit),
#                           torch.sigmoid(net.conc_drop3.p_logit)])
#     np.savetxt("pFMnistConDropr__" + str(i) + ".csv", res, delimiter=",")
#     enr = outofsample(net)
#     np.savetxt(X=enr, fname="pentroConcretefm" + str(i) + ".csv", delimiter=",")
#     torch.save(net.state_dict(), "ffmpConcrete" + str(i) + ".par")
#


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')



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

    net.eval()
    correct = 0
    obj = 500
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    # ots = np.zeros(obj, dtype=int)
    # dts = np.zeros((obj,5,784))

    count = 0
    spars = np.zeros(TEST_SAMPLES)
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

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


# %%

sns.set_style("dark")
# show(make_grid(dts.cpu()))
net.eval()
res, b = net(dts)
fmnist_outputs = res.max(1, keepdim=True)[1].detach().cpu().numpy()
fmnist_means = sigmoid(res.detach().cpu().numpy())
for _ in range(99):
    res, b = net(dts)
    fmnist_outputs = np.append(fmnist_outputs, res.max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)
    fmnist_means = fmnist_means + sigmoid(res.detach().cpu().numpy())
fmnist_means /= 100
sns.set_style("dark")
fig, ax = plt.subplots(30, 2, figsize=(10, 30), gridspec_kw={'width_ratios': [1, 8]})
j = -1
for i in range(30, obj, 2):
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

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


net.eval()
correct = 0
corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
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
            outputs[i], b = net.forward(data)
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
        # outputs[TEST_SAMPLES] = net(data, sample=True)
        # outputs[TEST_SAMPLES + 1] = net(data, sample=False)
        output = outputs[0:TEST_SAMPLES].mean(0)
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

def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


cdf(entropies.flatten())
np.savetxt(X=entropies.flatten(), fname="mnistfmnistdropout.csv", delimiter=",")

# %%


