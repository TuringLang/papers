using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
data {
  int D;
  int N;
  matrix[N, D] X;
  int<lower=0,upper=1> y[N];
}
parameters {
  vector[D] w;
}
model {
  w ~ normal(0, 1);
  y ~ bernoulli_logit_glm(X, 0.0, w);
}
"

step_size = 0.1
n_steps = 4

include("../infer_stan.jl")

;
