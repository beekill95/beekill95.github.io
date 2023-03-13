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
# Well, it's not technically from scratch, I'm going to use Numpy library to perform matrix calculation.
# In fact, I implemented [neural networks](https://github.com/beekill95/NumberRecognition)
# totally from scratch using C++ with just standard library before.
# That was a fun experience.
# Then why would I do it again?
# Well, because recently I had a homework,
# in which I had to derive the gradient descent update rules
# and implement the networks to solve the noise and signal separation problem.
# The thing was I struggled to derive the update rules.
# Therefore, I decided to write about this to commit it to my memory.
# If it's not, then I guess I still have a place to refer to it.

# %% [markdown]
# # Network Architecture
#
# Before deriving the gradient descent rule,
# we need to have a simple neural network.
# The network consists of 2 hidden layers (3 neurons and 5 neurons each),
# and an output layer with 2 neurons.

# %% [markdown]
# # Gradient Descent
# ## Derivative of the loss function wrt the output layer's parameters
# ## Derivative of the loss function wrt the second hidden layer's parameters

# %% [markdown]
# # Test
