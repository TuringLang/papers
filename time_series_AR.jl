
#Import Turing, Distributions, StatsBase, DataFrames and CSV
using Turing, Distributions, StatsBase, DataFrames, CSV

# Import MCMCChain, Plots and StatsPlots for visualizations and diagnostics
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Load in the shampoo dataset (can be downloaded from https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv)
df = CSV.read("shampoo.csv")
s = Float64[]
for ele in df[:Sales]
    push!(s, ele)
end
pyplot()
plot(s, reuse = false, title = "Shampoo dataset")
gui()

# Split into training and test sets. We will predict for the next 4 days using the data from the past 32 days
train_percentage = 0.9
s_train = s[1:floor(Int, train_percentage*length(s))]
N = length(s_train)

# Plot the training data
plot(s_train, reuse = false, title = "Train Data")
gui()

#Plot ACF and PACF plots. The PACF plot cuts off at k = 2, so we will have an AR(2) model for this dataset.
s1 = scatter([1, 2, 3, 4, 5], autocor(s, [1, 2, 3, 4, 5]), title = "ACF")
s2 = scatter([1, 2, 3, 4, 5], pacf(s, [1, 2, 3, 4, 5]), title = "PACF")
plot(s1, s2, layout = (2, 1), reuse = false)
gui()

#Defining the model
σ = 1
@model AR(x, N) = begin
    α ~ Normal(0,σ) 
    beta_1 ~ Uniform(-1, 1)
    beta_2 ~ Uniform(-1, 1)
    for t in 3:N
        μ = α + beta_1 * x[t-1] + beta_2 * x[t - 2] 
        x[t] ~ Normal(μ, 0.1) 
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sample using NUTS(n_iters::Int, n_adapts::Int, δ::Float64), where:
# n_iters::Int : The number of samples to pull.
# n_adapts::Int : The number of samples to use with adapatation.
# δ::Float64 : Target acceptance rate.

chain = sample(AR(s_train, N), NUTS(500, 200, 0.65) )

#Plotting the chain distribution of the sampled parameters and their values over the 500 iterations
#Note that roughly the first 50 samples are the warmup samples that we will remove later on
plot(chain, reuse = false, title = "Sampler Plot")
gui()

#Plotting the corner plot for the chain
corner(chain, reuse = false, title = "Corner Plot")
gui()

#Removing the warmup samples
chains_new = chain[50:500]

# Getting the mean values of the sampled parameters
beta_1 = mean(chains_new[:beta_1].value)
beta_2 = mean(chains_new[:beta_2].value)

#Obtaining the test data
s_test = s[N + 1:length(s)]

#Obtaining the predicted results using the AR(2) definition
s_pred = Float64[]
first_ele =  s_train[N]*beta_1 + s_train[N - 1]*beta_2 + rand(Normal(0,1))
push!(s_pred, first_ele)
second_ele = s_pred[1]*beta_1 + s_train[N]*beta_2 + rand(Normal(0,1))
push!(s_pred, second_ele)
for i=3:length(s_test)
    next_ele = s_pred[i - 1]*beta_1 + s_pred[i - 2]*beta_2 + rand(Normal(0,1))
    push!(s_pred, next_ele)
end

#Plotting the test and the predicted data for comparison
plot(s_test, reuse = false, title = "Predicted vs Test Comparison")
plot!(s_pred)
gui()

println("Press ENTER to exit")
read(stdin, Char)