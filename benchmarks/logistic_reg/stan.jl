using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int D;
  int N;
  matrix[D,N] X;
  int<lower=0,upper=1> y[N];
}
transformed data {
  matrix[N,D] XT;
  XT = X';
}
parameters {
  vector[D] w;
}
transformed parameters {
  vector[N] p;
  p = inv_logit(XT * w);
}
model {
  w ~ normal(0, 1);
  y ~ bernoulli(p);
}
"

step_size = 0.1
n_steps = 4

include("../infer_stan.jl")

;