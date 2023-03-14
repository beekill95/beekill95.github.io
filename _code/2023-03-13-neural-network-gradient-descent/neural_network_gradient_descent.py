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
#
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
#
# Next, I denote $\mathbf{X}^{(l)}$ as the input to the $l$-th layer.
# For instance, $\mathbf{X}^{(1)}$ is the input of the first hidden layer,
# and also the input to our neural network;
# and $\mathbf{X}^{(3)}$ is the input to the output layer.
# Here, unlike common convention,
# ...

# %% [markdown]
# # Gradient Descent
# ## Derivative of the loss function wrt the output layer's parameters
# ## Derivative of the loss function wrt the second hidden layer's parameters

# %% [markdown]
# # Test
