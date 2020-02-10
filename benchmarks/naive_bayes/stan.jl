using Random: seed!
seed!(1)

include("data.jl")

data = get_stan_data()

using CmdStan

const model_str = "
data {
  int C;
  int D;
  int N;
  matrix[N, D] image;
  int<lower=1,upper=C> label[N];
}
parameters {
  matrix[C, D] m;
}
model {
  for (d in 1:D) {
    target+= normal_lpdf(m[, d] | 0, 10);
  }

  for (d in 1:D) {
    target += normal_lpdf(image[, d] | m[label, d], 1);
  }
}
"

step_size = 0.1
n_steps = 4

include("../infer_stan.jl")

;
