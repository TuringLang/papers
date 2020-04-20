using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real m;
  real<lower=0> s;
}
model {
  m ~ normal(0, 1);
  s ~ normal(0, 2);
  y ~ normal(m, s);
}
"

step_size = 0.01
n_steps = 4

include("../infer_stan.jl")

;
