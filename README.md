# adversarial-learning

*Deep neural network(DNNs) are vulnerable to perturbations on images. Despite the high performance on clean datasets, we need to consider this shortcoming when deploying to real applications. This is the code I experimented with in adversarial learning.*

## Project Structure

Here is the structure of my project. Adversarial directory is my code of adversarial method. "networks" directory is my code of DNNs model. 
<details>
<summary>Expand to view</summary>
<pre><code>
adversarial-learning:.
│  .gitignore
│  examize.py
│  LICENSE
│  main.py
│  README.md
│  requirements.txt
│  tools.py
│
├─adversarial
│  │  attack.py
│  │  BlackPepper.py
│  │  Gaussian.py
│  │  PGD.py
│  │
│  └─__pycache__
├─Dataset
│  │  dataset.py
│  ├─cifar10
│  ├─MNIST
│  └─__pycache__
│          dataset.cpython-38.pyc
├─logs
├─networks
│  │  CNN.py
│  │  models.py
│  │  ResNet.py
│  │  wide_resnet.py
│  │  __init__.py
│  │
│  └─__pycache__
│          models.cpython-38.pyc
│          ResNet.cpython-38.pyc
│          wide_resnet.cpython-38.pyc
│          __init__.cpython-38.pyc
│
├─runs
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

## Details

