#Import Turing, Distributions and StatsBase
using Turing, Distributions, StatsBase

# Import MCMCChain, Plots and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Manually generate data
N = 500
s = zeros(N)
μ = 5
for i=3:N
    s[i] = μ + rand(Normal(0, 1)) + 0.5*rand(Normal(0, 1)) + 0.4*rand(Normal(0, 1))
end

pyplot()
plot(s, reuse = false, title = "Plot of the Time Series")
gui()

#Define the model

σ = 1
@model MA(x, N) = begin
    beta_1 ~ Uniform(-1, 1)
    beta_2 ~ Uniform(-1, 1)
    μ ~ Uniform(0, 10)
    for t in 3:N
        val = μ + rand(Normal(0,σ)) + beta_1 * rand(Normal(0, 1)) + beta_2 * rand(Normal(0, 1)) 
        x[t] ~ Normal(val, 1) 
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sample using HMC
chain = sample(MA(s, N), NUTS(500, 200, 0.65) )

# Print the summary of the sampled parameters
show(chain)

println("Plotting the chain distribution of the sampled parameters and their values over the 500 iterations")

plot(chain, reuse = false, title = "Sampler Plot (with warmup samples)")
gui()
corner(chain, reuse = false, title = "Corner Plot (with warmup samples)")
gui()

println("Note that roughly the first 50 samples are the warmup samples")
println("Removing these warmup samples...")

chain_new = chain[50:500]

plot(chain_new, reuse = false, title = "Sampler Plot (warmup samples removed)")
gui()
corner(chain_new, reuse = false, title = "Corner Plot (warmup samples removed")
gui()

println("Press ENTER to exit")
read(stdin, Char)