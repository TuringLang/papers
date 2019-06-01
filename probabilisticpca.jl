# Ported from Edward example: http://edwardlib.org/tutorials/probabilistic-pca

# This file shows how to prepare data and model for Probabilistic Principal Component Analysis using Turing

# Import required libraries
using Distributions, StatsPlots, Random

# Set a random seed.
Random.seed!(3)

# Generating the data
# We start off by creating a toy dataset.

function build_toy_dataset(N, D, K, sigma=1)
    
    x_train = Array{Real}(undef, D, N)
    w = rand(Normal(0.0, 2.0), D, K)
    z = rand(Normal(0.0, 1.0), K, N)
    mean = w*z
     
    for d in 1:D, n in 1:N
        x_train[d, n] = rand(Normal(mean[d, n], sigma))
    end
    
    return x_train
end


N = 5000  # number of data points
D = 2  # data dimensionality
K = 1  # latent dimensionality

x_train = build_toy_dataset(N, D, K)

# Next is the model definition

using Turing, MCMCChains

# The Probabilistic Principal Component Analysis model takes one arguments:
# * x: Set of data points for which to calculate gaussian latent variables


@model PPCA(x) = begin
    
    D, N = size(x)
    
    K = 1
    
    # Set principal axis prior
    w = Matrix{Real}(undef, D, K)   
    for d in 1:D, k in 1:K
        w[d,k] ~ Normal(0, 2)
    end
    
    # Set latent variable prior
    z = Matrix{Real}(undef, K, N)
    for k in 1:K, n in 1:N
        z[k,n] ~ Normal(0, 1)
    end
    
    mean = w*z
    
    # Set data prior
    for d in 1:D, n in 1:N
        x[d, n] ~ Normal(mean[d, n], 1)
    end
    
    return w
end

ppca_model = PPCA(x_train)

# Sampling from the posterior

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 1500
ϵ = 0.05
τ = 10

# Start sampling.
chain = mapreduce(c -> sample(ppca_model, HMC(iterations, ϵ, τ)), chainscat, 1:3)