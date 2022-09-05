from fileinput import filename
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from myutils.optimizer import get_scheduler
from networks.models import get_model
from myutils.tools import *
from Dataset.dataset import data
import os
import sys
from networks.ResNet import *
import argparse
from networks import *
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
use_cuda = torch.cuda.is_available()
from adversarial.attack import get_attack
from myutils.augmentation import mixup_criterion,mixup_data

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["MKL_NUM_THREADS"] = '4'
# os.environ["NUMEXPR_NUM_THREADS"] = '4'
# os.environ["OMP_NUM_THREADS"] = '4'

# initlize the code
parser = argparse.ArgumentParser(description='adversarial learning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='resnet18', type=str, help='select the model to use')
parser.add_argument('--load', default=None, type=str)
parser.add_argument('--depth', default=34, type=int, help='depth of the wide-resnet')
parser.add_argument('--widen_factor', default=10, type=int, help='widen factor for wide-resnet')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout for wide-resnet')
parser.add_argument('--dataset', default='cifar10', type=str, help='name of the dataset')
parser.add_argument('--parseval', '-p', default=False , action='store_true', help='whether to use parseval networks')
parser.add_argument('--num_epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size',type=int,default=64, help='batch_size of the network')
parser.add_argument('--debug','-d',action='store_true',default=False, help='enable debugging')
parser.add_argument('--criterion', default='cross_entropy', type=str, help='name of the criterion')
parser.add_argument('--lr_scheduler',default='exponential',type=str, help='name of the learning rate scheduler')
parser.add_argument('--weight_decay',default=5e-4,type=float, help='weight decay for regularization')
parser.add_argument('--attack', default=None, type=str, help='name of the attack function')
parser.add_argument('--eps', default=2/255, type=float, help='number of attack arguments')
parser.add_argument('--mode',default=0, type=int, help='mode: if is 0, then trian_epoch, if 1, then attack_train, if 2, then attack_test') # mode: if is 0, then trian_epoch, if 1, then attack_train, if 2, then attack_test
parser.add_argument('--save', default=None, type=str, help='save the model to disk')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for training')
parser.add_argument('--optimizer',default='sgd',type=str, help='name of the optimizer')
parser.add_argument('--mixup','-m', action='store_true', default=False, help='mixup augmentation')    
parser.add_argument('--onlytest','-o',action='store_true')
parser.add_argument('--alpha', default=1, type=float, help='mixup interpolation coefficient (default: 1)')
# parser.add_argument('--augumentation')
parser.add_argument('--transfer', default=0, type=int)
args = parser.parse_args()

# parsering
# self.batch_size = args.batch_size
# self.parseval = args.parserval
# self.depth = args.depth
# self.num_epochs = args.num_epochs
# self.dataset = args.dataset
# self.lr = args.lr
# self.dropout = args.dropout 

trainloader,testloader,num_classes, trainset, testset = data(args.model, args.dataset, args.batch_size, args.num_workers)

if args.load is not None:
    path = './checkpoints/' + args.load + '.pt'
    model = torch.load(path)
else:
    model = get_model(args, num_classes=num_classes)
# print(model)
print("building model...")

filename = "model" + args.model + "_mode" + str(args.mode)

if use_cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.attack is None:
    atk = None
else: 
    atk = get_attack(model, args.attack, args.eps)
    # print(atk)
    
print("loading tools...")
timer = timer()
logger = log(filename)
writer = writer(filename)
    

def info(args):
    logger.info("| running the code...")
    logger.info("running time: {}".format(timer.logtime()))
    logger.info('''Hyperparamter:
            model: {}, mode: {}, eps: {}, save: {}, optimizer: {}
            initilaze_learning_rate: {},  dataset: {}, lr_scheduler: {} 
            parseval: {},  num_epochs: {},  wide_factor: {}, num_workers: {}
            debug: {},  dropout: {},   criterion: {}, batch_size: {}
    '''.format(args.model, args.mode, args.eps, args.save, args.optimizer, args.lr, args.dataset, args.lr_scheduler, args.parseval, args.num_epochs, args.widen_factor, args.num_workers,
    args.debug, args.dropout, args.criterion,args.batch_size))
# information log in
info(args)



if args.optimizer == 'sgd':
    if args.parseval == False:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)
elif args.optimizer == 'l-bfgs':
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)

scheduler = get_scheduler(args, optimizer)

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
            if args.mode == 1:
                adv_imgs = atk(inputs, targets).cuda()
        elif args.mode == 1: 
            adv_imgs = atk(inputs, targets)
        
        if batch_idx == 0 and args.mode == 1:
            writer.images(adv_imgs[:4], epoch, 'images/adversarial')
            writer.images(inputs[:4], epoch, 'images/original')
        
        if args.mixup == True and epoch > 50:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
            
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            if args.mode == 1:
                outputs = model(adv_imgs)
            else:
                outputs = model(inputs)               # Forward Propagation
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        else: 
            optimizer.zero_grad()
            if args.mode == 1:
                outputs = model(adv_imgs)
            else:
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
        
def save():
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    save_point = './checkpoints/' + args.save + '.pt'
    torch.save(model, save_point)

def only_test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.to(device)
            targets =  targets.to(device)
            if atk is not None:
                adv_imgs = atk(inputs, targets).cuda()
        else: 
            if atk is not None:
                adv_imgs = atk(inputs, targets)
        
        if atk is not None:
            outputs = model(adv_imgs)
        else:
            outputs = model(inputs)
            
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print('|dataset: {}, parseval: {}, attack: {}, eps: {}, Acc@1: {},'.format(args.dataset, args.parseval, args.attack, args.eps, acc))
    return acc, test_loss

def test(epoch, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.to(device)
            targets =  targets.to(device)
            if args.mode == 2:
                adv_imgs = atk(inputs, targets).cuda()
        elif args.mode == 2:
            adv_imgs = atk(inputs, targets)
        
        if args.mode == 2:
            outputs = model(adv_imgs)
        else:
            outputs = model(inputs)
            
        loss = criterion(outputs, targets)
        if batch_idx == 0 and args.mode in [1, 2]:
            writer.images(adv_imgs[:4], epoch, 'images/adversarial')
            writer.images(inputs[:4], epoch, 'images/original')

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f test_Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    logger.info("\n| Validation Epoch #%d\t\t\tLoss: %.4f test_Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    if acc > best_acc and args.save is not None:
        save()
        best_acc = acc
    return acc, test_loss

if __name__ == "__main__":
    elapsed_time = 0

    if args.onlytest:
        only_test()
        exit(0)

    for epoch in range(1, args.num_epochs+1):
        start_time = time.time()
        
        best_acc = 0
        train_acc, train_loss = train_epoch(epoch)
        test_acc, test_loss = test(epoch, best_acc)

        if args.lr_scheduler:
            scheduler.step()
            
        if args.mode == 1 or args.mode == 2:
            writer.test_acc(test_acc, epoch)
            writer.train_acc(train_acc, epoch)
            writer.train_loss(train_loss, epoch)
            writer.test_loss(test_loss, epoch)       
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logger.info(timer.runtime())
        writer.close()

    logger.info("endding time: {}, successfully running".format(timer.filetime()))