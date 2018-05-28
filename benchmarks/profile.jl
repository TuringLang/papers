using HDF5, JLD, ProfileView

using Turing
setadbackend(:reverse_diff)
turnprogress(false)

include(splitdir(Base.@__DIR__)[1]*"/stan-models/lda-stan.data.jl")
include(splitdir(Base.@__DIR__)[1]*"/stan-models/lda.model.jl")

sample(ldamodel(data=ldastandata[1]), HMC(2, 0.025, 10))
Profile.clear()
@profile sample(ldamodel(data=ldastandata[1]), HMC(2000, 0.025, 10))

ProfileView.svgwrite("ldamodel.svg")