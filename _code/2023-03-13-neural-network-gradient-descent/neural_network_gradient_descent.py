# ---
# title: Neural Networks' Gradient Descent from Scratch
# layout: post
# use_math: true
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Well, it's not technically from scratch,
# I'm going to use Numpy library to perform matrix calculation.
# In fact, I implemented [a neural network](https://github.com/beekill95/NumberRecognition)
# totally from scratch using C++ with just standard library before.
# That was a fun experience.
# Then why would I do it again?
# Well, because recently I had a homework,
# in which I had to derive the gradient descent update rules
# and implement the networks to solve the noise and signal separation problem.
# The thing was I struggled to derive the rules.
# Therefore, I decided to write about this to commit it to my memory.
# If it's not, then I guess I still have a place to refer to it later.

# The game plan is:
# First, I'll introduce the network architecture along with the notations used.
# Then, I'll derive the gradients of layers' weights and biases;
# it's actually very simple!
# Alongside, I will also show how we can implement these gradients in Python.
# Finally, we tie it altogether by implementing a simple neural network for XOR dataset.
# Without further ado, let's get started!

# %% [markdown]
# # Network Architecture
#
# Before deriving the gradient descent rule,
# we need to have a simple neural network.
# The network consists of 2 hidden layers (3 neurons and 5 neurons each),
# and an output layer with 2 neurons.

# <div class="mermaid">
# <!--TODO: network architecture as described above-->
# </div>

# Before diving into the math,
# we should get ourselves familiar with the notations.
# First and foremost, following math convention,
# bold uppercase letters denote matrices, such as $\mathbf{X}$, $\mathbf{Y}$, etc.;
# while bold lowercase letters denote column vectors, such as $\mathbf{w}$,
# and normal letters denote scalar values, such as $\alpha$.

# Next, I denote $\mathbf{X}^{(l)}$ as the input to the $l$-th layer.
# For instance, $\mathbf{X}^{(1)}$ is the input of the first hidden layer,
# and also the input to our neural network;
# and $\mathbf{X}^{(3)}$ is the input to the output layer.
# Here, unlike common convention,
# each column in the input matrix $\mathbf{X}^{(1)}$ is a sample,
# and each row is a feature.

# TODO: include annotation
# $$
# \mathbf{X}^{(1)} = \begin{bmatrix}
#   \mathbf{x}^{(1)}_1 & \mathbf{x}^{(1)}_2 & \ldots & \mathbf{x}^{(1)}_N
# \end{bmarix},
# $$
# where $N$ is the number of input samples.

# Similarly, $\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$
# are the $l$-th layer's weight and bias.
# The weight and bias of the $i$-th neuron $i$ in the $l$-th layer
# are $\mathbf{w}^{(l,i)}$ and $b^{(l,i)}$, respectively.

# $$
# \mathbf{W}^{(3)} = \begin{bmatrix}
#   \mathbf{w}^{(3,1)} & \mathbf{w}^{(3,2)}
# \end{bmatrix},
# \mathbf{b}^{(3)} = \begin{bmatrix}
#   b^{(3,1)} \\
#   b^{(3,2)} \\
# \end{bmatrix}
# $$

# Furthermore, the linear output of the $l$-th layer is denoted as:
# $$
# \mathbf{Y}^{(l)} =
# \mathbf{W}^{(l)}^T \mathbf{X}^{(l)}
# + \mathbf{b}^{(l)} \mathbf{e}^T,
# $$
# where $\mathbf{e}$ is a vector of all 1 with appropriate
# length to make the matrix addition works.
#
# Finally, the activation output of a layer is denoted as
# $\mathbf{O}^{(l)} = g(\mathbf{Y}^{(l)})$.
# Thus, the output of the neural network is
# $$
# \mathbf{O}^{(3)}
# = g(\mathbf{Y}^{(3)})
# = \begin{bmatrix}
#   g(\mathbf{y}^{(3, 1)})^T \\
#   g(\mathbf{y}^{(3, 2)})^T \\
# \end{bmatrix}
# = \begin{bmatrix}
#   g(y^{(3, 1)}_1) & g(y^{(3, 1)}_2) & \ldots & g(y^{(3, 1)}_N) \\
#   g(y^{(3, 2)}_1) & g(y^{(3, 2)}_2) & \ldots & g(y^{(3, 2)}_N) \\
# \end{bmatrix}
# $$

# %% [markdown]
# # Gradient Descent
# ## Derivative of the loss function wrt the output layer's parameters
# ## Derivative of the loss function wrt the second hidden layer's parameters

# %% [markdown]
# # Test
