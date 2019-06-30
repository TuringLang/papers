
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
println("Loading the dataset")
df = CSV.read("shampoo.csv")
s = Float64[]
for ele in df[:Sales]
    push!(s, ele)
end
pyplot()
plot(s, reuse = false, title = "Shampoo dataset")
gui()

println("Split into training and test sets. We will predict for the next 4 days using the data from the past 32 days")
train_percentage = 0.9
s_train = s[1:floor(Int, train_percentage*length(s))]
N = length(s_train)

println("Plot the training data")
plot(s_train, reuse = false, title = "Train Data")
gui()

println("Plotting ACF and PACF plots") 
s1 = scatter([1, 2, 3, 4, 5], autocor(s, [1, 2, 3, 4, 5]), title = "ACF")
s2 = scatter([1, 2, 3, 4, 5], pacf(s, [1, 2, 3, 4, 5]), title = "PACF")
plot(s1, s2, layout = (2, 1), reuse = false)
gui()
println("The PACF plot cuts off at k = 2, so we will have an AR(2) model for this dataset")

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

println("Sampling using NUTS...")
chain = sample(AR(s_train, N), NUTS(500, 200, 0.65) )

println("Chain has been sampled; Now let us visualise it!")
plot(chain, reuse = false, title = "Sampler Plot")
gui()

#Plotting the corner plot for the chain
corner(chain, reuse = false, title = "Corner Plot")
gui()

println("Removing the warmup samples...")
chains_new = chain[50:500]
show(chains_new)

# Getting the mean values of the sampled parameters
beta_1 = mean(chains_new[:beta_1].value)
beta_2 = mean(chains_new[:beta_2].value)


println("Obtaining the test data")
s_test = s[N + 1:length(s)]

println("Obtaining the predicted results using the mean values of beta_1 and beta_2")
s_pred = Float64[]
first_ele =  s_train[N]*beta_1 + s_train[N - 1]*beta_2 + rand(Normal(0,1))
push!(s_pred, first_ele)
second_ele = s_pred[1]*beta_1 + s_train[N]*beta_2 + rand(Normal(0,1))
push!(s_pred, second_ele)
for i=3:length(s_test)
    next_ele = s_pred[i - 1]*beta_1 + s_pred[i - 2]*beta_2 + rand(Normal(0,1))
    push!(s_pred, next_ele)
end

println("Plotting the test and the predicted data for comparison")
plot(s_test, reuse = false, title = "Predicted vs Test Comparison", label = "Test")
plot!(s_pred, label = "Predicted")
gui()

println("Press ENTER to exit")
read(stdin, Char)