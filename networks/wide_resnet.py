import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def orthogonal_retraction(module, beta=0.002):
    with torch.no_grad():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, nn.Conv2d):
                weight_ = module.weight.data
                sz = weight_.shape
                weight_ = weight_.reshape(sz[0],-1)
                rows = list(range(module.weight.data.shape[0]))
            elif isinstance(module, nn.Linear):
                if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0], -1)
                    rows = list(range(module.weight.data.shape[0]))
                else:
                    rand_rows = np.random.permutation(module.weight.data.shape[0])
                    rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                    weight_ = module.weight.data[rows,:]
                    sz = weight_.shape
            module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out


def convex_constraint(module):
    with torch.no_grad():
        if isinstance(module, ConvexCombination):
            comb = module.comb.data
            alpha = torch.sort(comb, descending=True)[0]
            k = 1
            for j in range(1,module.n+1):
                if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                    k = j
                else:
                    break
            gamma = (torch.sum(alpha[:k]) - 1)/k
            module.comb.data -= gamma
            torch.relu_(module.comb.data)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, convex_combination=False):
        super(wide_basic, self).__init__()
        self.convex_combination = convex_combination
        if self.convex_combination:
            self.convex = ConvexCombination(2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        if not self.convex_combination:
            out += self.shortcut(x)
            return out
        else:
            return self.convex(out, self.shortcut(x))

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, convex_combination=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.convex_combination = convex_combination

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, convex_combination=self.convex_combination)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, convex_combination=self.convex_combination)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, convex_combination=self.convex_combination)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, convex_combination=False):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, convex_combination=convex_combination))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    print(net)
    y = net(torch.randn(1,3,32,32))

    print(y.size())
