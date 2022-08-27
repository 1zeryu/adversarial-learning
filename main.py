
from fileinput import filename
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
from tools import *
from Dataset.dataset import CIFAR, Mnist
import os
import sys
from networks.ResNet import *
import argparse
from networks import *
from torch.autograd import Variable
from torch.optim import lr_scheduler
use_cuda = torch.cuda.is_available()
from torchattacks import PGD, FGSM, CW

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["MKL_NUM_THREADS"] = '4'
# os.environ["NUMEXPR_NUM_THREADS"] = '4'
# os.environ["OMP_NUM_THREADS"] = '4'

# initlize the code
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--widen_factor', default=10, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--parseval', '-p', default=False , action='store_true')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--debug','-d',action='store_true',default=False)
parser.add_argument('--criterion', default='sgd', type=str)
parser.add_argument('--lr_scheduler',default='exponential',type=str)
parser.add_argument('--weight_decay',default=5e-4,type=float)
parser.add_argument('--attack', default=None, type=str)
parser.add_argument('--eps', default=2/255, type=float)
parser.add_argument('--mode',default=0, type=int) # mode: if is 0, then trian_epoch, if 1, then attack_train, if 2, then attack_test
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--num_workers', default=0, type=int)
# parser.add_argument('--augumentation')
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
        data = CIFAR(dataset)
        train_dataloader, test_dataloader, num_classes = data.dataloader(args.batch_size, args.num_workers)
        
    elif dataset == 'cifar100':
        data = CIFAR(dataset)
        train_dataloader, test_dataloader, num_classes = data.dataloader(args.batch_size, args.num_workers)

    elif dataset == 'mnist':
        data = Mnist()
        train_dataloader, test_dataloader, num_classes = data.dataloader(args.batch_size, args.num_workers)
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

filename = "model" + args.model + "_mode" + str(args.mode)

if use_cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.attack is None:
    pass
elif args.attack == 'pgd':
    atk = PGD(model, eps=args.eps)
elif args.attack == 'fgsm':
    atk = FGSM(model, eps=args.eps)
elif args.attack == 'cw':
    atk = CW(model, eps=args.eps)

print("loading tools...")
timer = timer()
logger = log(filename)
writer = writer(filename)
    

def info(args):
    logger.info("| running the code...")
    logger.info("running time: {}".format(timer.logtime()))
    logger.info('''Hyperparamter:
            model: {}, mode: {}, eps: {}
            initilaze_learning_rate: {},  dataset: {}, lr_scheduler: {} \n
            parseval: {},  num_epochs: {},  wide_factor: {}, \n
            debug: {},  dropout: {},   criterion: {}, batch_size: {}
    '''.format(args.model, args.mode, args.eps, args.lr, args.dataset, args.lr_scheduler, args.parseval, args.num_epochs, args.widen_factor,
    args.debug, args.dropout, args.criterion,args.batch_size))
# information log in
info(args)



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
    return acc, train_loss

    # Training
def attack_train(epoch):
    model.train()
    model.training = True
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.to(device)
            targets = targets.to(device)
            adv_imgs = atk(inputs, targets).cuda()
        if batch_idx == 0:
            writer.images(adv_imgs[:16], epoch, 'images/adversarial')
            writer.images(inputs[:16], epoch, 'images/original')
        optimizer.zero_grad()
        outputs = model(adv_imgs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
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
    return acc, train_loss
        
def save(state):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    save_point = './checkpoints/'
    torch.save(state, save_point+filename+'.pt')

def test(epoch, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
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
        if acc > best_acc and args.save:
            logger.info('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':model.module if use_cuda else model,
                    'acc':acc,
                    'epoch':epoch,
            }
            save(state)
            best_acc = acc
    return acc, test_loss

def attack_test(epoch, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.to(device)
            targets =  targets.to(device)
            adv_imgs = atk(inputs, targets).cuda()
        if batch_idx == 0:
            writer.images(adv_imgs[:16], epoch, 'images/adversarial')
            writer.images(inputs[:16], epoch, 'images/original')
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
    if acc > best_acc and args.save:
        logger.info('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':model.module if use_cuda else model,
                'acc':acc,
                'epoch':epoch,
        }
        save(state)
        best_acc = acc
    return acc, test_loss



if __name__ == "__main__":
    elapsed_time = 0
    for epoch in range(1, args.num_epochs+1):
        start_time = time.time()
        if args.mode == 0:
            best_acc = 0
            train_acc, train_loss = train_epoch(epoch)
            test_acc, test_loss = test(epoch, best_acc)
        elif args.mode == 1:
            best_acc = 0
            train_acc, train_loss = attack_train(epoch)
            test_acc, test_loss = test(epoch, best_acc)
        elif args.mode == 2:
            best_acc = 0
            train_acc, train_loss = train_epoch(epoch)
            test_acc, test_loss = attack_test(epoch, best_acc=best_acc)

        if args.lr_scheduler:
            scheduler.step()
        writer.test_acc(test_acc, epoch)
        writer.train_acc(train_acc, epoch)
        writer.train_loss(train_loss, epoch)
        writer.test_loss(test_loss, epoch)       
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logger.info('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
        writer.close()
    logger.info("endding time: {}, successfully running".format(timer.filetime()))
    logger.close()