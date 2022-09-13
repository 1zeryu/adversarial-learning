# adversarial-learning

*Deep neural network(DNNs) are vulnerable to perturbations on images. Despite the high performance on clean datasets, we need to consider this shortcoming when deploying to real applications. This is the code I experimented with in adversarial learning.*

## Project Structure

Here is the structure of my project. Adversarial directory is my code of adversarial method. "networks" directory is my code of DNNs model. 
<details>
<summary>Expand to view</summary>
<pre><code>
adversarial-learning
├─adversarial 
│  └─__pycache__
├─checkpoints
├─csv
├─Dataset
│  ├─cifar10
│  │  └─cifar-10-batches-py
│  ├─MNIST
│  │  └─raw
│  └─__pycache__
├─demo
├─example
├─logs
├─myutils
│  └─__pycache__
├─networks
│  └─__pycache__
├─runs
│  ├─220901190206modelpretrainedvit_mode0
│  ├─220911143121modellip_vit_mode0
│  └─220911143201modelvit_mode0
├─save
└─__pycache__
</code></pre>
</details>

## Usage

### clone

```bash
git clone https://github.com/1zeryu/adversarial-learning.git
```

### Requirements

```bash
pip install requirements.txt
```

### Run

You can understand some arguments in advance:

```bash
python main.py -h
```

## Attacks

* **Gaussian** 

* **FGSM**

Fast Gradient Sign Method(FGSM), based on the rational hypothesis that DNNs is sufficiently linear in nature, add one-order noise at a time to increase the loss. 
$$
x^{\prime}=x+\varepsilon \cdot \operatorname{sign}\left(\nabla_x J(x, y)\right)
$$


* **PGD**

 PGD, Project Gradient Descent, which can be seen as a replica of FSGM, is the iterative gradient attack method. PGD seek superb direction of attack by iterative without the linear hypothesis of FGSM.
$$
x_{t + 1}=\prod_{x + S}\left(x_t+\alpha \cdot \operatorname{sign}\left(\nabla_x J\left(x_t, y\right)\right)\right)
$$


## ![adversarial example](img/insert.svg)

## Lipschitz Defense

We use Lipschitz constraint in our model. 

