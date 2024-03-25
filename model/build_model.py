from model.lenet import LeNet
from model.model_resnet import resnet18, resnet34
from model.model_resnet_official import ResNet50
import torchvision.models as models
import torch.nn as nn
from model import unet, pspnet

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def build_model(args,config,train_loader):
    # choose different Neural network model for different args or inpu
    if args.model == 'lenet':
        netglob = LeNet().to(args.device)

    elif args.model == 'resnet18':
        netglob = resnet18(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet34':
        netglob = resnet34(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet50':
        netglob = ResNet50(pretrained=False)
        if args.pretrained:
            model = models.resnet50(pretrained=True)
            netglob.load_state_dict(model.state_dict())
        netglob.fc = nn.Linear(2048, args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'UNetResnet':
        netglob = get_instance(unet, 'arch', config, train_loader.dataset.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'NestedUnet':
        netglob = get_instance(unet, 'arch', config, train_loader.dataset.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'PSPNet':
        netglob = get_instance(pspnet, 'arch', config, train_loader.dataset.num_classes)
        netglob = netglob.to(args.device)
    
    else:
        exit('Error: unrecognized model')

    
    return netglob