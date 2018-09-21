using Distributions

M = 10
N = 4M

μs = 3 * [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]
xs = [rand(MvNormal(μs[1], [1.0, 1.0]), M)';
      rand(MvNormal(μs[2], [1.0, 1.0]), M)';
      rand(MvNormal(μs[3], [1.0, 1.0]), M)';
      rand(MvNormal(μs[4], [1.0, 1.0]), M)']
ts = [zeros(Int, 2M); ones(Int, 2M)]

bayes_nn_data = Dict(
  "N" => N,
  "xs" => xs,
  "ts" => ts)

# Save to file

using JLD

save("bayes-nn-data.jld", "data", bayes_nn_data)
