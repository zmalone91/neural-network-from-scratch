# Neural Networks: From Scratch to PyTorch

This repository contains implementations of neural networks using different approaches - from a detailed implementation built from scratch to streamlined PyTorch versions. The goal is to demonstrate how deep learning concepts translate between mathematical theory and modern frameworks.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Repository Overview

This project includes three different implementations of neural networks trained on the same moon-shaped classification dataset:

1. **From Scratch Implementation**: A detailed neural network built using only NumPy, implementing every component manually
2. **PyTorch Class-Based Implementation**: A neural network using PyTorch's object-oriented approach
3. **Simplified PyTorch Implementation**: An ultra-compact neural network using PyTorch's Sequential API

All implementations are designed to tackle the same binary classification problem, allowing for direct comparison of approaches.

## 🧠 Neural Network From Scratch

The from-scratch implementation builds a complete neural network using only NumPy, implementing all components manually:

### Key Features

- **Complete Implementation**: Forward propagation, backpropagation, weight updates, and various activation functions
- **Modular Design**: Separate components for different network operations
- **Visualization Tools**: Functions to visualize decision boundaries, weight distributions, and training progress
- **Educational Value**: Every mathematical operation is explicitly coded, showing exactly how neural networks work

### Major Components

- **Neuron Class**: Mathematical model of artificial neurons
- **Neural Network Class**: Implementation of multi-layer neural networks
- **Activation Functions**: Sigmoid, Tanh, ReLU, and Leaky ReLU with their derivatives
- **Loss Functions**: Mean Squared Error and Binary Cross-Entropy
- **Training Algorithms**: Mini-batch gradient descent with customizable parameters

This implementation makes explicit all the mathematical operations that happen "under the hood" in modern deep learning frameworks.

## ⚡ PyTorch Implementations

### Class-Based PyTorch Neural Network

This implementation uses PyTorch's object-oriented approach to build a neural network:

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        
        # Create layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```

### Simplified PyTorch Implementation

This ultra-compact implementation uses PyTorch's Sequential API to create a neural network in just a few lines:

```python
# One-line model definition
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Simple training loop
for epoch in range(300):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📊 Visualizations

The repository includes various visualizations to understand neural network behavior:

- **Decision Boundary Evolution**: How the model's classification boundary changes during training
- **Weight Distribution**: How weight values are distributed and change over time
- **Training Metrics**: Loss and accuracy curves for both training and validation data

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- PyTorch 1.9+
- scikit-learn
- seaborn

## 🔑 Key Concepts Illustrated

### From Neural Network Theory to Practice

This repository demonstrates the transition from mathematical neural network theory to practical implementation, showing:

1. How matrix operations translate to PyTorch's layer abstractions
2. How the chain rule of calculus becomes PyTorch's autograd
3. How weight initialization strategies affect learning
4. How activation functions influence model behavior
5. How optimizers improve upon basic gradient descent

### Comparison of Implementations

| Feature | From Scratch | PyTorch Class | PyTorch Simple |
|---------|-------------|--------------|---------------|
| Lines of Code | ~500 | ~150 | ~50 |
| Implementation Complexity | High | Medium | Low |
| Performance | Moderate | Excellent | Excellent |
| Educational Value | Highest | High | Moderate |
| Flexibility | Custom | High | Limited |
| Integration with Ecosystem | None | Full | Full |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgments & 📚 References

The implementations draw from numerous resources in the deep learning community:

Neural Network Fundamentals and Backpropagation
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/

Activation Functions
- Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 807-814).
- Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. In Proc. ICML (Vol. 30, No. 1, p. 3).
- Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289.

Weight Initialization
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (pp. 249-256).
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1026-1034).
- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120.

Optimization Algorithms
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 26-31.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. In International Conference on Machine Learning (pp. 1139-1147).

Loss Functions and Metrics
- De Boer, P. T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. (2005). A tutorial on the cross-entropy method. Annals of Operations Research, 134(1), 19-67.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980-2988).

Regularization Techniques
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (pp. 448-456).
- Krogh, A., & Hertz, J. A. (1992). A simple weight decay can improve generalization. In Advances in Neural Information Processing Systems (pp. 950-957).

Learning Rate Scheduling
- Smith, L. N. (2017). Cyclical learning rates for training neural networks. In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
- Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.

Neural Network Visualization
- Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In European Conference on Computer Vision (pp. 818-833). Springer, Cham.
- Olah, C., Mordvintsev, A., & Schubert, L. (2017). Feature visualization. Distill, 2(11), e7.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (pp. 618-626).

Extended Neural Network Architectures (CNN, RNN)
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (pp. 1097-1105).
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

Textbooks and Comprehensive Resources
- Nielsen, M. (2015). Neural Networks and Deep Learning. Determination Press. http://neuralnetworksanddeeplearning.com/
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. O'Reilly Media.
- Chollet, F. (2018). Deep Learning with Python. Manning Publications.

Code and Implementation Inspiration
- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/

- The visualization techniques were influenced by publications on distill.pub
