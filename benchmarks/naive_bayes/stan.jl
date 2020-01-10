using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int C;
  int D;
  int N;
  matrix[D,N] image;
  int<lower=1,upper=C> label[N];
}
parameters {
  matrix[D,C] m;
}
model {
  for (c in 1:C)
    for (d in 1:D)
        m[d,c] ~ normal(0, 10);
      
  for (n in 1:N)
    for (d in 1:D)
        image[d,n] ~ normal(m[d,label[n]], 1);
}
"

alg = CmdStan.Hmc(
    CmdStan.Static(0.4),
    CmdStan.diag_e(),
    0.1,
    0.0,
)

include("../infer_stan.jl")

;