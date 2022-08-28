import torchattacks as tk

def get_attack(model, attack, eps):
    attack = attack.upper()
    if attack == 'PGD':
        atk = tk.PGD(model, eps=eps)
    elif attack == 'FGSM':
        atk = tk.FGSM(model, eps=eps)
    elif attack == 'CW':
        atk = tk.CW(model)
    return atk