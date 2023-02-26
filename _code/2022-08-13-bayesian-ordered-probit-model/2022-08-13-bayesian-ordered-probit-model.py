# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: title,author,layout,use_math
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   layout: post
#   title: Bayesian Ordered Probit Model
#   use_math: true
# ---

# %% [markdown]
# # Ordinal Data in Real World
#
# Ordinal data is a categorical data type,
# where the variables have ordered categories.
# However, unlike metric data type,
# where we can easily calculate the distance between variables,
# we couldn't do the same with ordinal variables.

# There are many examples of ordinal data type in real world.
# For instance, in many e-commerce websites such as Amazon, eBay, etc.,
# ordinal data appeared as ratings given to a product by customers.
# Or in schools or educational institutes,
# letter grades given by teachers to evaluate the performance of students
# on exams or tests can also be considered as ordinal data.
# Or if we ask a bunch of people filling in a survey form by expressing
# their aggreement on a scale from 1 to 7
# (with 1 being "totally disagree" and 7 being "totally agree")
# about various statements.

# It is also important to note that in the examples above,
# even though the labels of the categories are numbers,
# they are not metric variables.
# Thus, these numbers should be thought as text labels,
# representing an ascending or descending order.
# In fact, many research papers model oridinal data as metric data
# and get wrong results (TODO: citation here).

# Naturally, we would want to ask how customers, teachers, subjects come up
# with such ratings, evaluations, and answers, respectively.
# Moreover, we would want compare between products or students
# to find out which product or student is the best.
# Or in the case of the survey form,
# we want to find which statements are widely believed,
# or which statements are controversial.

# %% [markdown]
# # Modelling

# %% [markdown]
# # Implementation in Numpyro
# ## Data
# ## Model
# ## Results

# %% [markdown]
# # Wrapups

# %% [markdown]
# # References
# - Chapter 23, Doing Bayesian Data Analysis, 2nd Edition, John K. Kruschke,
# https://sites.google.com/site/doingbayesiandataanalysis/
# - https://betanalpha.github.io/assets/case_studies/ordinal_regression.html
# - https://num.pyro.ai/en/stable/tutorials/ordinal_regression.html
