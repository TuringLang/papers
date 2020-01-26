using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

# See:
# https://ecosang.github.io/blog/study/dirichlet-process-with-stan/

# TODO



step_size = 0.01
n_steps = 4

include("../infer_stan.jl")

;
