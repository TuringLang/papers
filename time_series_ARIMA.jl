
# Import Turing, Distributions, StatsBase, DataFrames and CSV, HypothesisTests and LinearAlgebra
using Turing, Distributions, StatsBase, DataFrames, CSV, HypothesisTests, LinearAlgebra

# Import MCMCChain, Plots and StatsPlots
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Load in the dataset downloaded from https://github.com/inertia7/timeSeries_sp500_R/blob/master/data/data_master_1.csv
#MIT License
println("Loading the dataset")
df = CSV.read("../data/data_master_1.csv")
df[:sp_500]

s = df.sp_500
pyplot()
plot(s, reuse = false, title = "Plot of the complete data")
gui()

# Split into training and test sets. 
println("Split into training and test sets. We will predict for the next 4 days using the data from the past 32 days")
train_percentage = 0.95
s_train = s[1:floor(Int, train_percentage*length(s))]
N = length(s_train)

# Plot the training data
println("Plot the training data")
plot(s_train, reuse = false, title = "Plot of the training data")
gui()

ADFTest(s_train, Symbol("constant"), 5)

println("Plot the differenced series")
s_diff = diff(s_train)
plot(s_diff, reuse = false, title = "Plot of the differenced series")
gui()

ADFTest(s_diff, Symbol("constant"), 5)

#Plot ACF and PACF plots
println("Plotting ACF and PACF plots")
total_lags = 20
s1 = scatter(collect(1:total_lags), autocor(s_diff, collect(1:total_lags)), title = "ACF", ylim = [-0.3,0.5])
s2 = scatter(collect(1:total_lags), pacf(s_diff, collect(1:total_lags)), title = "PACF", ylim = [-0.3,0.5])
plot(s1, s2, layout = (2, 1), reuse = false)
gui()
println("From the ACF and PACF plots, both ARIMA(0, 1, 1) and ARIMA(1, 1, 0) seem plausible")

@model ARIMA110(x) = begin
    T = length(x)
    μ ~ Uniform(-10, 10)
    ϕ ~ Uniform(-1, 1)
    for t in 3:T
        val = μ +                      # Drift term.
              x[t-1] +                 # ARIMA(0,1,0) portion.
              ϕ * (x[t-1] - x[t-2]) # ARIMA(1,0,0) portion.
        x[t] ~ Normal(val, 1)
    end
end

println("Sampling ARIMA(1, 1, 0) using NUTS...")
chain_ARIMA110 = sample(ARIMA110(s_train), NUTS(10000, 200, 0.6) )

plot(chain_ARIMA110, reuse = false, title = "Sampler plot for ARIMA(1, 1, 0) model")
gui()

@model ARIMA011(x) = begin
    T = length(x)

    # Set up error vector.
    ϵ = Vector(undef, T)
    x_hat = Vector(undef, T)

    θ ~ Uniform(-5, 5)

    # Treat the first x_hat as a parameter to estimate.
    x_hat[1] ~ Normal(550, 220)
    ϵ[1] = x[1] - x_hat[1]

    for t in 2:T
        # Predicted value for x.
        x_hat[t] = x[t-1] - θ * ϵ[t-1]
        # Calculate observed error.
        ϵ[t] = x[t] - x_hat[t]
        # Observe likelihood.
        x[t] ~ Normal(x_hat[t], 1)
    end
end

println("Sampling ARIMA(0, 0, 1) using NUTS...")
chain_ARIMA011 = sample(ARIMA011(s_train), NUTS(10000, 1000, 0.6) )

plot(chain_ARIMA011, reuse = false, title = "Sampler plot for ARIMA(0, 1, 1) model")
gui()

# ARIMA 110 AIC calculation

function calculate_aic_ARIMA110(β::Float64, μ:: Float64, σ::Float64, s::Array{Float64, 1})
    T = length(s)
    ϵ = Vector(undef, T)
    s_pred = Vector(undef, T)
    
    s_pred[1], s_pred[2] = s[1], s[2]
    ϵ[1], ϵ[2] = 0.0, 0.0 
    for t in 3:T
        s_pred[t] = μ +                      
              s_pred[t-1] +                 
              β * (s_pred[t-1] - s_pred[t-2]) 
        ϵ[t] = s_pred[t] - s[t]
    end
    log_likelihood = (-(T - 1)/2)*2*π*σ^2 - (1/σ^2)*sum(ϵ.^2) - π*σ^2/(1 - β^2) - ((s[1] - μ/(1 - β))^2)/(2*σ^2/(1 - β^2))
    aic = -2*log_likelihood + 2
    return aic
end

# ARIMA 011 AIC calculation

function calculate_aic_ARIMA011(β::Float64, σ::Float64, s::Array{Float64, 1})
    T = length(s)

    ϵ = Vector(undef, T)
    s_pred = Vector(undef, T)

    s_pred[1] = s[1]
    ϵ[1] = 0.0
    for t in 2:T
        s_pred[t] = s[t-1] - β * ϵ[t-1]
        ϵ[t] = s[t] - s_pred[t]
    end
    log_likelihood = -(T/2)*log(2*π) - (T/2)*log(σ^2) + (1/2*σ^2)*sum(ϵ.^2)  
    aic = -2*log_likelihood + 1
    return aic
end

aicARIMA011 = calculate_aic_ARIMA011(mean(chain_ARIMA011[:θ].value), 1.0, s_train)
println("AIC value for ARIMA011: ")
println(aicARIMA011)

aicARIMA110 = calculate_aic_ARIMA110(mean(chain_ARIMA110[:ϕ].value), mean(chain_ARIMA110[:μ].value), 1.0, s_train)
println("AIC value for ARIMA110: ")
println(aicARIMA110)

println("Press ENTER to exit")
read(stdin, Char)
