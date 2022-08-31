from networks.ResNet import ResNet18, ResNet34, ResNet50
from networks.wide_resnet import Wide_ResNet
import torch.nn as nn
import torchvision
from networks.ViT import vit

def get_model(args, num_classes):
    if args.model == 'wide-resnet':
        model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, convex_combination=args.parseval)
    
    elif args.model == 'resnet18':
        model = ResNet18(num_classes)
        
    elif args.model == 'resnet50':
        model = ResNet50(num_classes)
        
    elif args.model == 'resnet34':
        model = ResNet34(num_classes)
        
    elif args.model == 'vit':
        model = vit(32, 4, num_classes=num_classes, dropout=args.dropout)
    
    elif args.model == 'pretrainedvit':
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes))
    elif args.model == 'swint':
        model = torchvision.models.swin_t(weights='IMAGENET1K_V1')
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes))
        
    return model