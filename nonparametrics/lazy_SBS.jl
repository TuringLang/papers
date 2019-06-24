using Turing, Turing.RandomMeasures
using RDatasets, Plots
using Statistics
# Data
data = [-2,2,-1.5,1.5]

# Base distribution
mu_0 = mean(data)
sigma_0 = 4
sigma_1 = 0.5
tau0 = 1/sigma_0^2
tau1 = 1/sigma_1^2

# DP parameters
alpha = 0.25

# size-biased sampling process
@model sbsimm(y, rpm) = begin
    # Base distribution.
    H = Normal(mu_0, sigma_0)

    # Latent assignments.
    N = length(y)
    z = tzeros(Int, N)

    # locations of Gaussians
    x = tzeros(Float64, N)
    
    # probability weights 
    J = tzeros(Float64, N)
    
    # assignments of observations
    z = tzeros(Int, N)

    k = 0
    surplus = 1.0
    for i in 1:N
        ps = vcat(J[1:k], surplus)
        z[i] ~ Categorical(ps)
        if z[i] > k
            k = k + 1
            J[k] ~ SizeBiasedSamplingProcess(rpm, surplus)
            x[k] ~ H
            surplus -= J[k]
        end
        y[i] ~ Normal(x[z[i]], sigma_1)
    end
end

rpm = DirichletProcess(alpha)

sampler = SMC(10)
mf = sbsimm(data, rpm)

# Compute empirical posterior distribution over partitions
samples = sample(mf, sampler)