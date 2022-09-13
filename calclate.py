def Params(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))


from networks.ResNet import ResNet18, ResNet34
from networks.wide_resnet import Wide_ResNet
from networks.ViT import vit, lip_vit

if __name__ == "__main__":
    model = Wide_ResNet(22, 10, 0.3, 10, convex_combination=True)

    Params(model)