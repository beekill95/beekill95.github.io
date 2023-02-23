# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: title,author,layout,use_math
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   layout: post
#   title: Ordered Probit Model - Part 1 - Modelling
#   use_math: true
# ---

# Ordering data are often encountered in real-life.
# You will certainly encounter these data in e-commerce websites
# like Amazon, eBay, etc.,
# in which products are rated by customers.
# Thus, a natural question arises:
# given products' ratings, how can we know which products are better?
#
# Similarly, ordered data are usually arised in surveys.
# Questions in these surveys are often in the form:
# in the scale of 1 to 5, with 1 is very bad and 5 is very good,
# how do you rate your happiness level today?
# Then, from the given answer, we want to know the true happiness level of the person
