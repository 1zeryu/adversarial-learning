import torchattacks as tk

from adversarial.Gaussian import GaussianNoise

def get_attack(model, attack, eps):
    attack = attack.upper()
    if attack == 'PGD':
        atk = tk.PGD(model, eps=eps)
    elif attack == 'FGSM':
        atk = tk.FGSM(model, eps=eps)
    elif attack == 'CW':
        atk = tk.CW(model)
    elif attack == 'gaussian':
        atk = GaussianNoise(eps)
    # elif attack == 'l-bfgs':
    #     atk = tk.LBFGS(model, eps=eps)
    return atk