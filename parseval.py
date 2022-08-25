import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F





if __name__ == "__main__":
    pass

    # import torch.optim as optim
    # net = ConvexCombination(10)
    # opt = optim.SGD(net.parameters(), lr=0.0001)
    # xs = [ torch.rand(2,2) for _ in range(10)]
    # for i in range(10):
    #     xs[i].requires_grad = True
    # y = net(*xs)
    # loss = y.sum()
    # loss.backward()
    # opt.step()
    # # print(net.comb.grad)
    # convex_constraint(net)


    # import torch.optim as optim
    #
    # x = torch.rand(4,3,10,10)
    # # net = nn.Conv2d(3,64,5)
    # net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(300, 20)
    # )
    #
    # opt = optim.SGD(net.parameters(), lr=0.0001)
    # y = net(x)
    # opt.zero_grad()
    # loss = y.sum()
    # loss.backward()
    # opt.step()
    # orthogonal_retraction(net[1])
