
from fileinput import filename
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
from tools import *
from Dataset.dataset import getCIFAR
import os
import sys
from networks.ResNet import *
import argparse
from networks import *
from torch.autograd import Variable
from torch.optim import lr_scheduler
use_cuda = torch.cuda.is_available()

import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'

# initlize the code
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--model', default='wide-resnet', type=str)
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--widen_factor', default=10, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--parseval', '-p', default=False , action='store_true')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--debug','-d',action='store_true',default=False)
parser.add_argument('--criterion', default='sgd', type=str)
parser.add_argument('--lr_scheduler',default=None,type=str)
parser.add_argument('--weight_decay',default=5e-4,type=float)
args = parser.parse_args()

# parsering
# self.batch_size = args.batch_size
# self.parseval = args.parserval
# self.depth = args.depth
# self.num_epochs = args.num_epochs
# self.dataset = args.dataset
# self.lr = args.lr
# self.dropout = args.dropout 
def dataset(args):
    dataset = args.dataset
    if dataset == 'cifar10':
        data = getCIFAR(dataset)
        train_dataloader, test_dataloader, num_classes = data.dataloader(args.batch_size)
        
    if dataset == 'cifar100':
        data = getCIFAR(dataset)
        train_dataloader, test_dataloader, num_classes = data.dataloader(args.batch_size)
    print("loading data...")
    return train_dataloader, test_dataloader, num_classes

trainloader,testloader,num_classes = dataset(args)
if args.model == 'wide-resnet':
    model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, convex_combination=args.parseval)
elif args.model == 'resnet18':
    model = ResNet18(num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes)
elif args.model == 'resnet34':
    model = ResNet34(num_classes)
print("building model...")

filename = "model" + args.model + "_epochs" + str(args.num_epochs) + '_args.criterion' \
                +args.criterion + "_parseval" + str(args.parseval)


print("loading tools...")
timer = timer()
logger = log(timer.filetime() + filename)
writer = writer(timer.filetime() + filename)
    

def info(self, args):
    self.logger.info("| running the code...")
    self.logger.info("running time: {}".format(timer.logtime()))
    self.logger.info('''Hyperparamter:
            model: {},
            initilaze_learning_rate: {},  dataset: {}, lr_scheduler: {} \n
            parseval: {},  num_epochs: {},  wide_factor: {}, \n
            debug: {},  dropout: {},   criterion: {}, batch_size: {}
    '''.format(args.model, args.lr, args.dataset, args.lr_scheduler, args.parseval, args.num_epochs, args.widen_factor,
    args.debug, args.dropout, args.criterion,args.batch_size))
# information log in
logger.info(args)



if args.parseval:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

if args.lr_scheduler == 'lambda1':
    lambda1 = lambda epoch:np.sin(epoch) / (epoch + 1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
elif args.lr_scheduler == 'step':
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)
elif args.lr_scheduler == 'exponential':
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(epoch):
    model.train()
    model.training = True
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        if args.parseval:
            for m in model.modules():
                orthogonal_retraction(m)
                convex_constraint(m)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch %3d, Data: %3d \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, batch_idx,loss.item(), 100.*correct/total))
        sys.stdout.flush()
    logger.info('| Epoch %3d, Data: %3d \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, batch_idx,loss.item(), 100.*correct/total))
    return acc

    # Training
best_acc = 0
def test(epoch, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f test_Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        logger.info("\n| Validation Epoch #%d\t\t\tLoss: %.4f test_Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        if acc > best_acc:
            logger.info('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':model.module if use_cuda else model,
                    'acc':acc,
                    'epoch':epoch,
            }
            save(state)
            best_acc = acc
    return acc

def save(state):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    save_point = './checkpoints/'
    torch.save(state, save_point+filename+'.pt')

if __name__ == "__main__":
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    elapsed_time = 0
    for epoch in range(1, args.num_epochs+1):
        start_time = time.time()

        train_acc = train_epoch(epoch)
        test_acc = test(epoch, best_acc)
        if args.lr_scheduler:
            scheduler.step()
        writer.test_acc(test_acc, epoch)
        writer.train_acc(train_acc, epoch)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logger.info('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))