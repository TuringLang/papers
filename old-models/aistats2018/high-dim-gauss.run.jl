using Turing

include(splitdir(Base.@__DIR__)[1]*"/aistats2018/high-dim-gauss.data.jl")
include(splitdir(Base.@__DIR__)[1]*"/aistats2018/high-dim-gauss.model.jl")

turnprogress(false)

mf = high_dim_gauss(data=hdgdata[1])
chn = sample(mf, HMC(1000, 0.05, 5))