# Ref: https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.stan

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

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
  phi ~ uniform(-1, 1);
  sigma ~ cauchy(0, 5);
  mu ~ cauchy(0, 10);  
  h[1] ~ normal(mu, sigma / sqrt(1 - phi * phi));
  for (t in 2:T)
    h[t] ~ normal(mu + phi * (h[t - 1] -  mu), sigma);
  for (t in 1:T)
    y[t] ~ normal(0, exp(h[t] / 2));
}
"

alg = CmdStan.Hmc(
    CmdStan.Static(0.004),
    CmdStan.diag_e(),
    0.001,
    0.0,
)

include("../infer_stan.jl")

;