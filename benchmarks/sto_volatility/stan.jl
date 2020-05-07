using DrWatson
@quickactivate "TuringExamples"

# Ref: https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.stan

using Random: seed!
seed!(1)

include("data.jl")

data = get_data(500)

const model_str = "
data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}
parameters {
  real mu;                     // mean log volatility
  real<lower=-1,upper=1> phi;  // persistence of volatility
  real<lower=0> sigma;         // white noise shock scale
  vector[T] h;                 // log volatility at time t
}
model {
  mu ~ cauchy(0, 10);
  phi ~ uniform(-1, 1);
  sigma ~ cauchy(0, 5);
  h[1] ~ normal(mu, sigma / sqrt(1 - phi * phi));
  for (t in 2:T)
    h[t] ~ normal(mu + phi * (h[t - 1] -  mu), sigma);
  y ~ normal(0, exp(h / 2));
}
"

step_size = 0.0002
n_steps = 4

include("../infer_stan.jl")

;
