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
parameters {
  vector[D] w;
}
model {
  target += normal_lpdf(w | 0, 1);
  target += bernoulli_logit_glm_lpmf(y | X, 1.0, w);
}
"

step_size = 0.1
n_steps = 4

include("../infer_stan.jl")

;
