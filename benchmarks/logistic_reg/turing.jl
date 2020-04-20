using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using LazyArrays, ReverseDiff, Memoization, Turing

lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))
safelogistic(x::T) where {T} = logistic(x) * (1 - 2 * eps(T)) + eps(T)
@model logistic_reg(X, y) = begin
    D, N = size(X)
    w ~ filldist(Normal(0, 1), D)
    y ~ arraydist(lazyarray(x -> Bernoulli(safelogistic(x)), X' * w))
end

model = logistic_reg(data["X"], data["y"])

step_size = 0.1
n_steps = 4
test_zygote = false
test_tracker = true

include("../infer_turing.jl")

;
