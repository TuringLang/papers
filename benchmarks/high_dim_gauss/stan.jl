using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int D;
}
parameters {
  vector[D] m;
}
model {
  m ~ normal(0, 1);
}
"

step_size = 0.1
n_steps = 4

include("../infer_stan.jl")

;
