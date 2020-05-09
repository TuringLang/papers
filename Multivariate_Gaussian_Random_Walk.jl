# Ported from https://docs.pymc.io/notebooks/MvGaussianRandomWalk_demo.html

# Import packages

using Turing, Distributions, MCMCChains
using LinearAlgebra, Plots

using Random
Random.seed!(12);

Turing.turnprogress(false);


# Define cumulative sum function

function cumsum(X)
    N, D = size(X)
    for j in 2:D
        X[:,j] = X[:, j] + X[:, j-1]
    end
    return X
end


# Generating data

D = 3
N = 300
sections = 5
period = div(N, sections)

Sigma_a = randn(D, D)
Sigma_a = transpose(Sigma_a)*Sigma_a
L_a = cholesky(Sigma_a).L

Sigma_b = randn(D, D)
Sigma_b = transpose(Sigma_b)*Sigma_b
L_b = cholesky(Sigma_b).L

# Gaussian Random Walk
alpha = transpose(cumsum(L_a*randn(D, sections)))
beta = transpose(cumsum(L_b*randn(D, sections)))
sigma = 0.1

t = reshape(Vector(collect(0:N-1))/N, (300, 1))
alpha = repeat(alpha, period)
beta = repeat(beta, period)
y = alpha + beta.*t .+ sigma*randn(N, 1);


# Plot the data

plot(t, y)
