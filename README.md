# GradLab - A Toy Deep Learning Framework

GradLab is a minimal, educational deep learning framework inspired by [PyTorch](https://github.com/pytorch/pytorch), [Tinygrad](https://github.com/tinygrad/tinygrad), [MiniTorch](https://github.com/minitorch/minitorch), and Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd). Its purpose is to help students, hobbyists, and anyone curious about neural networks explore the **mathematical concepts** behind them while learning how deep learning frameworks are designed and structured.

The idea was to provide a path for everyone's first (including my own) **baby steps into the fundamentals of neural networks**: building models, computing gradients with backpropagation, implementing optimisers, and training simple networks — all without the complexity of fully-featured libraries. It's a toy version of PyTorch that's low-level, readable, and approachable. By exploring the codebase of my toy dl framework, I hope you gain some insight into the inner-workings of neural networks, but also appreciate the **system design** behind frameworks like PyTorch.

---

## Project Structure

- **core/tensor.py** – Implements the `Value` (Thank you Andrej!), `Tensor` class, supporting basic arithmetic, ReLU, and simple autograd
- **core/nn.py** – Contains core layers (`Linear`, `ReLU`, `Sequential`) and prepares the groundwork for CNNs, RNNs, and more 
- **core/optim.py** – Implements gradient descent-based optimisers like `SGD` and `Adam`
- **tests/** - Runs through a series of unit tests to ensure that the classes, methods and functions worked as intended
- **demo_MLP.py** – Example script demonstrating the training and testing of a MLP model on the MNIST dataset
- **demo_CNN.py** – Example script demonstrating the training and testing of a CNN model on the FashionMNIST dataset (more demos may be added in the future)
- **More to come in the future** - demo_RNN, demo_Transformer

---

## Getting Started

1. Clone the repository:  
```bash
git clone <https://github.com/AI-I224/gradlab>
cd gradlab
```

2. Create virtual environment and install dependencies:

```bash
- m venv gradlab/.venv
source .venv/bin/activate
pip install -r requirements.txt
```

The virtual environment can be deactivated and deleted using the following commands:

```bash
deactivate
rm -rf gradlab/.venv
```

---

## Running Tests

1. To run the full unit test suite:  
```bash
python -m pytest
```

2. To run a specific unit test:  
```bash
python -m pytest tests/testfilename.py
```

---

## References

This framework draws inspiration from:

* [Micrograd](https://github.com/karpathy/micrograd) – for autograd engine design, `Value` class and `nn.py` layout
* [Tinygrad](https://github.com/tinygrad/tinygrad) and [MiniTorch](https://github.com/minitorch/minitorch) – for minimalistic, functional deep learning design
* [PyTorch](https://github.com/pytorch/pytorch) – for overall architecture, modularity, and naming conventions

The aim is **not to compete with production frameworks**, but to create a playground for understanding neural networks and the design principles behind them.

---

## Future Plans

* Implementing a simple Transformer model
* Adding more demos for RNNs and Transformers
* Improving documentation and explanations for beginners

---

## License

This project is MIT licensed — feel free to explore, modify, and learn.