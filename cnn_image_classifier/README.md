# README #

### Convolutional Neural Networks from scratch without any Neural Network Libraries

This small project was an attempt to understand the intricacies and the mathematics behind the backpropagation for 
convolutional layers in a better way.

A very crude version of CNN was implemented using mostly just `numpy` and tested on a binary classification dataset derived from a subset of the MNIST dataset with just 2 classes. A 97% accuracy score was obtained on the test set of the subset of MNIST dataset used.

The network has one convolutional layer followed by a `tanh activation layer` with other options like `ReLU` also available. This is followed by a single fully connected layer which in turn is followed by a sigmoid actvation layer. The network was trained using `Stochastic Gradient Descent`.



