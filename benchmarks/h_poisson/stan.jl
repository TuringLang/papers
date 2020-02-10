using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int N;
  int y[N];
  int Ns;
  int idx[N];
  matrix[1, N] x;
}
parameters {
  real a0;
  vector[Ns] a0s;
  vector[1] a1;
  real<lower=0> a0_sig;
}
model {
  vector[N] alpha;
  a0 ~ normal(0, 10);
  a1 ~ normal(0, 1);
  a0_sig ~ cauchy(0, 1);
  a0s ~ normal(0, a0_sig);
  y ~ poisson_log(alpha);
  target += poisson_log_glm_lpmf(y | x, a0 + a0s[idx], a1);
}
"

step_size = 0.01
n_steps = 4

include("../infer_stan.jl")

;
