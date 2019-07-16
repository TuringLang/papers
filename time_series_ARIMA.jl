
#Import Turing, Distributions, StatsBase, DataFrames and CSV
using Turing, Distributions, StatsBase, DataFrames, CSV

# Import MCMCChain, Plots and StatsPlots
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Load in the dataset downloaded from https://github.com/inertia7/timeSeries_sp500_R/blob/master/data/data_master_1.csv
#MIT License
df = CSV.read("../data/DowJones.csv")
df[:sp_500]
# plot(df[:Date], df[Symbol("Adj Close")])

s = Float64[]
for ele in df[:sp_500]
# for ele in df[Symbol("Adj Close")]
    push!(s, ele)
end
pyplot()
plot(s)

# Split into training and test sets. We will predict for the next 4 days using the data from the past 32 days
train_percentage = 0.95
s_train = s[1:floor(Int, train_percentage*length(s))]
N = length(s_train)

# Plot the training data
plot(s_train)

s_diff = diff(s_train)
N_diff = length(s_diff)
plot(s_diff)

#Plot ACF and PACF plots
total_lags = 10
s1 = scatter(collect(1:total_lags), autocor(s_diff, collect(1:total_lags)), title = "ACF")
s2 = scatter(collect(1:total_lags), pacf(s_diff, collect(1:total_lags)), title = "PACF")
plot(s1, s2, layout = (2, 1))
# The PACF plot cuts off at k = 2, so we will have an AR(2) model for this dataset.

#Defining the model
σ = 1
@model ARIMA110(x, N) = begin
    beta_1 ~ Uniform(-1, 1)
    for t in 3:N
        μ = rand(Normal(0, 1)) + beta_1 * x[t-1] 
        x[t] ~ Normal(μ, 1) 
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sample using NUTS
chain_ARIMA110 = sample(ARIMA110(s_diff, N_diff), NUTS(500, 200, 0.65) )

plot(chain_ARIMA110)

#Define the model

σ = 1
@model ARIMA011(x, N) = begin
    beta_1 ~ Uniform(-1, 1)
    μ ~ Uniform(0, 10)
    for t in 3:N
        val = μ + rand(Normal(0,σ)) + beta_1 * rand(Normal(0, 1)) 
        x[t] ~ Normal(val, 1) 
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sampling using NUTS
chain_ARIMA011 = sample(ARIMA011(s_diff, N_diff), NUTS(500, 200, 0.65) )

plot(chain_ARIMA011)

# ARIMA 011 WAIC calculation

sampling_size = 500
lpppd = 0.0
p = 0.0
for i = 1:N
    likelihood = Float64[]
    for sample = 1:sampling_size
        beta_1_value = chain_ARIMA011[:beta_1][sample] 
#         mu_value = chain_ARIMA011[:mu][sample,i,:]
#         dist = Normal(mu_value + eta_value[1]*tau_value, sigma[i])
        push!(likelihood, pdf(dist, y[i]) )
    end
    
    # Adding the contribution of the current observation in lpppd
    log_mean = log(mean(likelihood))
    lpppd += log_mean
    
    #Calculating the p_waic value
    log_likelihood = log.(likelihood)
    var_likelihood = var(log_likelihood)
    p += var_likelihood
end

#Applying the above mentioned equation to get the final WAIC value
waic = -2*(lpppd - p);
