# Ref: https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.stan
# Ref: https://mc-stan.org/docs/2_22/stan-users-guide/stochastic-volatility-models.html

using Random: seed!
seed!(1)

include("data.jl")

data = get_data(500)

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
  vector[T] h_std;             // std log volatility at time t
}
transformed parameters {
  vector[T] h = h_std * sigma;  // now h ~ normal(0, sigma)
  h[1] /= sqrt(1 - phi * phi);  // rescale h[1]
  h += mu;
  for (t in 2:T)
    h[t] += phi * (h[t-1] - mu);
}
model {
  phi ~ uniform(-1, 1);
  sigma ~ cauchy(0, 5);
  mu ~ cauchy(0, 10);  
  h_std ~ std_normal();
  y ~ normal(0, exp(h / 2));
}
"

step_size = 0.0002
n_steps = 4

include("../infer_stan.jl")

;