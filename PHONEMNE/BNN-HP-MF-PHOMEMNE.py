#matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
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
N = 3500
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


def reparametrize(mu, logvar, cuda=False, sampling=True):
    if sampling:
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu


# -------------------------------------------------------
# LINEAR LAYER
# -------------------------------------------------------

class LinearGroupNJ(Module):
    """Fully Connected Group Normal-Jeffrey's layer (aka Group Variational Dropout).
    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, clip_var=None):

        super(LinearGroupNJ, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.clip_var = clip_var
        self.deterministic = False  # flag is used for compressed inference
        # trainable params according to Eq.(6)
        # dropout params
        self.z_mu = Parameter(torch.Tensor(in_features))
        self.z_logvar = Parameter(torch.Tensor(in_features))  # = z_mu^2 * alpha
        # weight params
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))

        # init params either random or with pretrained net
        self.reset_parameters(init_weight, init_bias)

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # numerical stability param
        self.epsilon = 1e-8

    def reset_parameters(self, init_weight, init_bias):
        # init means
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        self.z_mu.data.normal_(1, 1e-2)

        if init_weight is not None:
            self.weight_mu.data = torch.Tensor(init_weight)
        else:
            self.weight_mu.data.normal_(0, stdv)

        if init_bias is not None:
            self.bias_mu.data = torch.Tensor(init_bias)
        else:
            self.bias_mu.data.fill_(0)

        # init logvars
        self.z_logvar.data.normal_(-9, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        if self.clip_var:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def get_log_dropout_rates(self):
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + self.epsilon)
        return log_alpha

    def compute_posterior_params(self):
        weight_var, z_var = self.weight_logvar.exp(), self.z_logvar.exp()
        self.post_weight_var = self.z_mu.pow(2) * weight_var + z_var * self.weight_mu.pow(2) + z_var * weight_var
        self.post_weight_mu = self.weight_mu * self.z_mu
        return self.post_weight_mu, self.post_weight_var

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.linear(x, self.post_weight_mu, self.bias_mu)

        batch_size = x.size()[0]
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1), self.z_logvar.repeat(batch_size, 1), sampling=self.training,
                          cuda=DEVICE)

        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        xz = x * z
        mu_activations = F.linear(xz, self.weight_mu, self.bias_mu)
        var_activations = F.linear(xz.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp())

        return reparametrize(mu_activations, var_activations.log(), sampling=self.training, cuda=DEVICE)

    def kl_divergence(self):
        # KL(q(z)||p(z))
        # we use the kl divergence approximation given by [2] Eq.(14)
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(k1 * self.sigmoid(k2 + k3 * log_alpha) - 0.5 * self.softplus(-log_alpha) - k1)

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = -0.5 * self.weight_logvar + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        # KL bias
        KLD_element = -0.5 * self.bias_logvar + 0.5 * (self.bias_logvar.exp() + self.bias_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        return KLD

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# -------------------------------------------------------
# CONVOLUTIONAL LAYER
# -------------------------------------------------------

class _ConvNdGroupNJ(Module):
    """Convolutional Group Normal-Jeffrey's layers (aka Group Variational Dropout).
    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, init_weight, init_bias, cuda=False, clip_var=None):
        super(_ConvNdGroupNJ, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        self.cuda = cuda
        self.clip_var = clip_var
        self.deterministic = False  # flag is used for compressed inference

        if transposed:
            self.weight_mu = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_logvar = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight_mu = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_logvar = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

        self.bias_mu = Parameter(torch.Tensor(out_channels))
        self.bias_logvar = Parameter(torch.Tensor(out_channels))

        self.z_mu = Parameter(torch.Tensor(self.out_channels))
        self.z_logvar = Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters(init_weight, init_bias)

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        # numerical stability param
        self.epsilon = 1e-8

    def reset_parameters(self, init_weight, init_bias):
        # init means
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        # init means
        if init_weight is not None:
            self.weight_mu.data = init_weight
        else:
            self.weight_mu.data.uniform_(-stdv, stdv)

        if init_bias is not None:
            self.bias_mu.data = init_bias
        else:
            self.bias_mu.data.fill_(0)

        # inti z
        self.z_mu.data.normal_(1, 1e-2)

        # init logvars
        self.z_logvar.data.normal_(-9, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        if self.clip_var:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def get_log_dropout_rates(self):
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + self.epsilon)
        return log_alpha

    def compute_posterior_params(self):
        weight_var, z_var = self.weight_logvar.exp(), self.z_logvar.exp()
        self.post_weight_var = self.z_mu.pow(2) * weight_var + z_var * self.weight_mu.pow(2) + z_var * weight_var
        self.post_weight_mu = self.weight_mu * self.z_mu
        return self.post_weight_mu, self.post_weight_var

    def kl_divergence(self):
        # KL(q(z)||p(z))
        # we use the kl divergence approximation given by [2] Eq.(14)
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(k1 * self.sigmoid(k2 + k3 * log_alpha) - 0.5 * self.softplus(-log_alpha) - k1)

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = - 0.5 * self.weight_logvar + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        # KL bias
        KLD_element = - 0.5 * self.bias_logvar + 0.5 * (self.bias_logvar.exp() + self.bias_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        return KLD

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv1dGroupNJ(_ConvNdGroupNJ):
    r"""
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cuda=False, init_weight=None, init_bias=None, clip_var=None):
        kernel_size = utils._single(kernel_size)
        stride = utils._single(stride)
        padding = utils._single(padding)
        dilation = utils._single(dilation)

        super(Conv1dGroupNJ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, utils._pair(0), groups, bias, init_weight, init_bias, cuda, clip_var)

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.conv1d(x, self.post_weight_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        batch_size = x.size()[0]
        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        mu_activations = F.conv1d(x, self.weight_mu, self.bias_mu, self.stride,
                                  self.padding, self.dilation, self.groups)

        var_activations = F.conv1d(x.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp(), self.stride,
                                   self.padding, self.dilation, self.groups)
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1, 1), self.z_logvar.repeat(batch_size, 1, 1),
                          sampling=self.training, cuda=self.cuda)
        z = z[:, :, None]

        return reparametrize(mu_activations * z, (var_activations * z.pow(2)).log(), sampling=self.training,
                             cuda=self.cuda)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Conv2dGroupNJ(_ConvNdGroupNJ):
    r"""
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cuda=False, init_weight=None, init_bias=None, clip_var=None):
        kernel_size = utils._pair(kernel_size)
        stride = utils._pair(stride)
        padding = utils._pair(padding)
        dilation = utils._pair(dilation)

        super(Conv2dGroupNJ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, utils._pair(0), groups, bias, init_weight, init_bias, cuda, clip_var)

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.conv2d(x, self.post_weight_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        batch_size = x.size()[0]
        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        mu_activations = F.conv2d(x, self.weight_mu, self.bias_mu, self.stride,
                                  self.padding, self.dilation, self.groups)

        var_activations = F.conv2d(x.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp(), self.stride,
                                   self.padding, self.dilation, self.groups)
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1), self.z_logvar.repeat(batch_size, 1),
                          sampling=self.training, cuda=self.cuda)
        z = z[:, :, None, None]

        return reparametrize(mu_activations * z, (var_activations * z.pow(2)).log(), sampling=self.training,
                             cuda=self.cuda)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Conv3dGroupNJ(_ConvNdGroupNJ):
    r"""
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cuda=False, init_weight=None, init_bias=None, clip_var=None):
        kernel_size = utils._triple(kernel_size)
        stride = utils._triple(stride)
        padding = utils._triple(padding)
        dilation = utils.triple(dilation)

        super(Conv3dGroupNJ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, utils._pair(0), groups, bias, init_weight, init_bias, cuda, clip_var)

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.conv3d(x, self.post_weight_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        batch_size = x.size()[0]
        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        mu_activations = F.conv3d(x, self.weight_mu, self.bias_mu, self.stride,
                                  self.padding, self.dilation, self.groups)

        var_weights = self.weight_logvar.exp()
        var_activations = F.conv3d(x.pow(2), var_weights, self.bias_logvar.exp(), self.stride,
                                   self.padding, self.dilation, self.groups)
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1, 1, 1, 1), self.z_logvar.repeat(batch_size, 1, 1, 1, 1),
                          sampling=self.training, cuda=self.cuda)
        z = z[:, :, None, None, None]

        return reparametrize(mu_activations * z, (var_activations * z.pow(2)).log(), sampling=self.training,
                             cuda=self.cuda)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# build a simple MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # activation
        self.relu = nn.ReLU()
        # layers
        self.fc1 = LinearGroupNJ(256, 400, clip_var=0.04)
        self.fc2 = LinearGroupNJ(400, 600)
        self.fc3 = LinearGroupNJ(600, 5)
        # layers including kl_divergence
        self.kl_list = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = x.view(-1, 256)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def get_masks(self, thresholds):
        weight_masks = []
        mask = None
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if mask is None:
                log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                mask = log_alpha < threshold
            else:
                mask = np.copy(next_mask)
            try:
                log_alpha = layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                next_mask = log_alpha < thresholds[i + 1]
            except:
                # must be the last mask
                next_mask = np.ones(5)

            weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            weight_masks.append(weight_mask.astype(np.float))
        return weight_masks

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD


discrimination_loss = nn.functional.cross_entropy


def objective(output, target, kl_divergence):
    discrimination_error = discrimination_loss(output, target)
    variational_bound = discrimination_error + kl_divergence / N
    # variational_bound = variational_bound.to.DEVICE()
    return variational_bound


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
        output = net(data)
        loss = objective(output, target, net.kl_divergence())
        loss.backward()
        optimizer.step()
        for layer in net.kl_list:
            layer.clip_variances()
    print(epoch + 1)
    print(loss)


def posttrain(net, optimizer, epoch,i,masks,layers):
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
        output = net(data)
        loss = objective(output, target, net.kl_divergence())
        loss.backward()
        optimizer.step()
        for layer, mask in zip(layers, masks):
            layer.post_weight_mu.data = layer.post_weight_mu.data.mul(torch.Tensor(mask).to(DEVICE))
            layer.post_weight_var.data = layer.post_weight_var.data.mul(torch.Tensor(mask).to(DEVICE))
        for layer in net.kl_list:
            layer.clip_variances()
    print(epoch+1)
    print(loss)


def test_ensembleolf():
    net.train()
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
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, False)
            outputs[TEST_SAMPLES] = net(data, True)
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


def unit_round_off(t=23):
    """
    :param t:
        number significand bits
    :return:
        unit round off based on nearest interpolation, for reference see [1]
    """
    return 0.5 * 2. ** (1. - t)


SIGNIFICANT_BIT_PRECISION = [unit_round_off(t=i + 1) for i in range(23)]


def float_precision(x):
    out = np.sum([x < sbp for sbp in SIGNIFICANT_BIT_PRECISION])
    return out


def float_precisions(X, dist_fun, layer=1):
    X = X.flatten()
    out = [float_precision(2 * x) for x in X]
    out = np.ceil(dist_fun(out))
    return out


def special_round(input, significant_bit):
    delta = unit_round_off(t=significant_bit)
    rounded = np.floor(input / delta + 0.5)
    rounded = rounded * delta
    return rounded


def fast_infernce_weights(w, exponent_bit, significant_bit):
    return special_round(w, significant_bit)


def compress_matrix(x):
    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B, C * D)
        # remove non-necessary filters and rows
        x = x[:, (x != 0).any(axis=0)]
        x = x[(x != 0).any(axis=1), :]
    else:
        # remove unnecessary rows, columns
        x = x[(x != 0).any(axis=1), :]
        x = x[:, (x != 0).any(axis=0)]
    return x


def extract_pruned_params(layers, masks):
    post_weight_mus = []
    post_weight_vars = []

    for i, (layer, mask) in enumerate(zip(layers, masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_var = post_weight_var.cpu().data.numpy()
        post_weight_mu = post_weight_mu.cpu().data.numpy()
        # apply mask to mus and variances
        post_weight_mu = post_weight_mu * mask
        post_weight_var = post_weight_var * mask

        post_weight_mus.append(post_weight_mu)
        post_weight_vars.append(post_weight_var)

    return post_weight_mus, post_weight_vars


# -------------------------------------------------------
#  Compression rates (fast inference scenario)
# -------------------------------------------------------


def _compute_compression_rate(vars, in_precision=32., dist_fun=lambda x: np.max(x), overflow=10e38):
    # compute in  number of bits occupied by the original architecture
    sizes = [v.size for v in vars]
    nb_weights = float(np.sum(sizes))
    IN_BITS = in_precision * nb_weights
    # prune architecture
    vars = [compress_matrix(v) for v in vars]
    sizes = [v.size for v in vars]
    # compute
    significant_bits = [float_precisions(v, dist_fun, layer=k + 1) for k, v in enumerate(vars)]
    exponent_bit = np.ceil(np.log2(np.log2(overflow) + 1.) + 1.)
    total_bits = [1. + exponent_bit + sb for sb in significant_bits]
    OUT_BITS = np.sum(np.asarray(sizes) * np.asarray(total_bits))
    return nb_weights / np.sum(sizes), IN_BITS / OUT_BITS, significant_bits, exponent_bit


def compute_compression_rate(layers, masks):
    # reduce architecture
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    # compute overflow level based on maximum weight
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    # compute compression rate
    CR_architecture, CR_fast_inference, _, _ = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x),
                                                                         overflow=overflow)
    print("Compressing the architecture will degrease the model by a factor of %.1f." % (CR_architecture))
    print("Making use of weight uncertainty can reduce the model by a factor of %.1f." % (CR_fast_inference))
    return CR_architecture, CR_fast_inference


def compute_reduced_weights(layers, masks):
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    _, _, significant_bits, exponent_bits = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x),
                                                                      overflow=overflow)
    weights = [fast_infernce_weights(weight_mu, exponent_bits, significant_bit) for weight_mu, significant_bit in
               zip(weight_mus, significant_bits)]
    return weights


# %%
#
# i = 1
# torch.manual_seed(i)
# net = Net().to(DEVICE)
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
# for epoch in range(epochs):
#     train(net, optimizer, epoch, i)
#
# def cdf(x, plot=True, *args, **kwargs):
#     x  = sorted(x)
#     y = np.arange(len(x)) / len(x)
#     return plt.plot(x, y, *args, **kwargs) if plot else (x, y)
# def sigmoid(x):
#         return (1 / (1 + np.exp(-x)))
# %%
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

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
            # print(target)
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                net.training = True
                net.fc1.training = True
                net.fc2.training = True
                net.fc3.training = True
                net.fc1.deterministic = False
                net.fc2.deterministic = False
                net.fc3.deterministic = False
                outputs[i] = net(data)
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
            net.fc1.compute_posterior_params()
            net.fc2.compute_posterior_params()
            net.fc3.compute_posterior_params()
            net.training = False
            net.fc1.training = False
            net.fc2.training = False
            net.fc3.training = False
            net.fc1.deterministic = True
            net.fc2.deterministic = True
            net.fc3.deterministic = True
            outputs[TEST_SAMPLES] = net(data)
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
        if (cases3 == 0): cases3 = 1
        corrects = np.append(corrects, correct3 / cases3)
        corrects = np.append(corrects, cases3)

        return corrects


# test_ensemble()


# %%



# %%

# make inference on 10 networks
for i in range(5, 10):
    print(i)
    torch.manual_seed(i)
    net: Net = Net().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train(net, optimizer, epoch, i)


    res = test_ensemble()

    print(res)
    np.savetxt("soundhorsepfull_" + str(i) + ".csv", res, delimiter=",")
    layers = [net.fc1, net.fc2, net.fc3]
    thresholds = [-2.8, -3., -5.]
    crs = compute_compression_rate(layers, net.get_masks(thresholds))

    print("Test error after with reduced bit precision:")
    masks = net.get_masks(thresholds)
    weights = compute_reduced_weights(layers, masks)
    for layer, weight, mask in zip(layers, weights, masks):
        layer.post_weight_mu.data = torch.Tensor(weight).to(DEVICE)
        layer.post_weight_var.data = layer.post_weight_var.data.mul(torch.Tensor(mask).to(DEVICE))
    #for layer in layers: layer.deterministic = True
    res = test_ensemble()
    res = np.append(res,crs)

    np.savetxt("soundhorseprunded_" + str(i) + ".csv", res, delimiter=",")
    for layer in layers: layer.deterministic = False
    for epoch in range(50):
        posttrain(net, optimizer, epoch, i, layers=layers, masks=masks)
    #for layer in layers: layer.deterministic = True
    res = test_ensemble()
    torch.save(net.state_dict(), "horseffmreducedpost" + str(i) + ".par")
    np.savetxt("ptsoundhorseprunded_" + str(i) + ".csv", res, delimiter=",")
