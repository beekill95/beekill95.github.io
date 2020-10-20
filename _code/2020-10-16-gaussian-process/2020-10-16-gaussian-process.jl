#' ---
#' title: A Note about Gaussian Processes
#' layout: post
#' use_math: true
#' ---

#' I have heard the term Gaussian processes many times but never quite understand how it actually works,
#' even after many posts and Youtube videos.
#' Today, I decided to spend the time to research and explain the topic based on my understanding,
#' because as someone has said, if you cannot write it down, then you don't understand it at all.
#' So the game plan for this post would be: explaining the math, then figuring out the intuition behind Gaussian processes,
#' and finally applying Gaussian processes to a toy problem.

#' # The Math
#' ## Bayesian Inference
#' 
#' Before diving into today's topic, it's worth reviewing Bayesian inference as it is the foundation of Gaussian process.
#' As the name suggested, Bayesian inference uses Bayes' theorem to update probability ...
#'
#' Usually, in Bayesian inference, the data is thought to come from a function $f$, which is parameterized by $\boldsymbol{\theta}$:
#' $\mathcal{D} = f(\boldsymbol{\theta})$.
#' Then, according to Bayes' theorem, one can derive the original parameters by the following equation:
#'
#' $$
#' P(\boldsymbol{\theta} | \mathcal{D}) =
#'     \frac{
#'         \stackrel{\text{likelihood}}{P(\mathcal{D} | \boldsymbol{\theta})}
#'         \stackrel{\text{prior}}{P(\boldsymbol{\theta})}
#'     }{\underset{\text{evidence}}{P(\mathcal{D})}}
#' $$
#'
#' ...
#' The _likelihood_, denoted by $P(\mathcal{D} | \boldsymbol{\theta})$, tells us how likely it is to see the observed data points
#' when the parameters are $\boldsymbol{\theta}$.
#' The _prior_, denoted by $P(\boldsymbol{\theta})$, is ...
#' The _evidence_, denoted by $P(\mathcal{D})$, shows how probable it is to observed those data points.
#' It is calculated by integrating over all possible parameters $\boldsymbol{\theta}$, which is often intractable
#' and ...
#' 
#' Bayesian inference is a very powerful approach to machine learning.
#' Many machine learning algorithms can be viewed under the light of Bayesian methods,
#' such as neural network, which is an interesting topic for later posts.
#' Now, we are ready to uncover the math behind Gaussian Processes.
#'
#' ## Gaussian Processes
#'
#' This is something

using Plots
import Random

# Set seed for random generator to get the same behavior whenever the script runs.
Random.seed!(1995)

"Generate some datapoints (x, y) with x is in the domain (-6, 6), and y = sin(x)"
function generate_points(nb_samples::Int; noise_level=0.0)
    x = (2 .* rand(Float64, nb_samples) .- 1) .* 6
    y = sin.(x) + randn(nb_samples) * noise_level
    return x, y
end

"""Square exponential covariance function"""
function squared_exponential_covariance(x, y; σ=1.0, ℓ=1.0)
    return σ^2 * exp(-(x - y)^2 / (2 * ℓ^2))
end

"""Calculate covariance matrix using the given kernel function"""
function covariance_matrix(x::Array{Float64,1}; noise_level=0.0, kernel_func=squared_exponential_covariance)
    length, = size(x)
    C = Array{Float64}(undef, length, length)

    for i in 1:length
        for j in 1:length
            C[i, j] = kernel_func(x[i], x[j]) + (i == j ? noise_level : 0)
        end
    end

    return C
end

"""
Perform regression on new data points given the observed data using Gaussian Process.
Returns a list of prediction (mean, variance) of each point.

t = y + noise
y = ∑w * ϕ(x)
"""
function gaussian_process(predicts::Array{Float64,1}; targets::Array{Float64,1}, observed::Array{Float64,1}, noise_level=0.0, kernel_func=squared_exponential_covariance)
    Cn = covariance_matrix(observed, kernel_func=kernel_func, noise_level=noise_level)
    inv_Cn = inv(Cn)

    length, = size(predicts)
    predictions = []

    for predict in predicts
        k = [kernel_func(predict, obs) for obs in observed]
        kappa = kernel_func(predict, predict) + noise_level

        mean = transpose(k) * inv_Cn * targets
        variance = kappa - transpose(k) * inv_Cn * k

        push!(predictions, (mean, variance))
    end

    return predictions
end

# Generate training points
x, y = generate_points(8, noise_level=1.0)
# Generate test points
xtest, ytest = let
    x = -8:0.1:8
    y = sin.(x)

    x, y
end

predictions = gaussian_process(collect(xtest), targets=y, observed=x, noise_level=1.0)

prediction_mean = first.(predictions)
prediction_std = sqrt.(last.(predictions))

# Plot the results.
plot(x, y, xlab="x", ylab="f(x)", seriestype=:scatter, label="train", xlims=(-8, 8), markersize=4, markeralpha=1)
plot!(xtest, ytest, seriestype=:scatter, label="test", markersize=3, markeralpha=0.8)
plot!(xtest, prediction_mean, yerror=prediction_std, seriestype=:scatter, label="prediction mean", markersize=4, markeralpha=1)

