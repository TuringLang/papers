using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
data {
  int N;
  int y[N];
  int Ns;
  int idx[N];
  real x[N];
}
parameters {
  real a0;
  vector[Ns] a0s;
  real a1;
  real<lower=0> a0_sig;
}
model {
  vector[N] alpha;
  a0 ~ normal(0, 10);
  a1 ~ normal(0, 1);
  a0_sig ~ cauchy(0, 1);
  a0s ~ normal(0, a0_sig);
  for(i in 1:N) alpha[i] = a0 + a0s[idx[i]] + a1 * x[i];
  y ~ poisson_log(alpha);
}
"

step_size = 0.001
n_steps = 4

include("../infer_stan.jl")

;
