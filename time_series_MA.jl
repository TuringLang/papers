#Import Turing, Distributions and StatsBase
using Turing, Distributions, StatsBase

# Import MCMCChain, Plots and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

println("Generating data...")
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
    delta_1 ~ Normal(0, 1)
    delta_2 ~ Normal(0, 1)
    for t in 3:N
        val = μ + beta_1 * delta_1 + beta_2 * delta_2 
        x[t] ~ Normal(val, 1) 
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

println("Sampling using NUTS...")
# NUTS(n_iters::Int, n_adapts::Int, δ::Float64), where:
# n_iters::Int : The number of samples to pull.
# n_adapts::Int : The number of samples to use with adapatation.
# δ::Float64 : Target acceptance rate.
chain = sample(MA(s, N), NUTS(500, 200, 0.65) )

println("Summary of the sampled chain:")
show(chain)

println("Plotting the chain distribution of the sampled parameters and their values over the 500 iterations")

plot(chain, reuse = false, title = "Sampler Plot (with warmup samples)")
gui()
corner(chain, reuse = false, title = "Corner Plot (with warmup samples)")
gui()

println("Note that roughly the first 50 samples are the warmup samples")
println("Removing these warmup samples...")

chain_new = chain[50:500]

println("Summary of the new chain:")
show(chain_new)
plot(chain_new, reuse = false, title = "Sampler Plot (warmup samples removed)")
gui()
corner(chain_new, reuse = false, title = "Corner Plot (warmup samples removed")
gui()

println("Press ENTER to exit")
read(stdin, Char)