import torchattacks as tk

class Attacks(object):
    def __init__(self, model, attack, eps):
#         atks = [
#     FGSM(model, eps=8/255),
#     BIM(model, eps=8/255, alpha=2/255, steps=100),
#     RFGSM(model, eps=8/255, alpha=2/255, steps=100),
#     CW(model, c=1, lr=0.01, steps=100, kappa=0),
#     PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
#     PGDL2(model, eps=1, alpha=0.2, steps=100),
#     EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
#     FFGSM(model, eps=8/255, alpha=10/255),
#     TPGD(model, eps=8/255, alpha=2/255, steps=100),
#     MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
#     VANILA(model),
#     GN(model, std=0.1),
#     APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
#     APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
#     APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
#     FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
#     FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
#     Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
#     AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
#     OnePixel(model, pixels=5, inf_batch=50),
#     DeepFool(model, steps=100),
#     DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
# ]
        attack = attack.lower()
        if attack == 'pgd':
            self.atk = tk.PGD(model, eps=eps)
        elif attack == 'fgsm':
            self.atk = tk.FGSM(model, eps=eps)
        elif attack == 'cw':
            self.atk = tk.CW(model, eps=eps)
            
    
    def __call__(self, images, labels):
        return self.atk(images, labels) 