using Turing

@model quick() = begin
  v ~ Normal(0, 1)
end

sample(quick(), HMC(16, 0.4, 2))

using Plots

plot(linspace(1,3,49), exp(linspace(1,3,49)), leg=false, ticks=nothing,
     annotations=(2, 10, text("Welcome to Turing.jl demo!", :bottom, 29)))
srand(12352)
