using Turing, Turing.RandomMeasures
using RDatasets, Plots
using Statistics

# -- load dataset --
iris = dataset("datasets", "iris")

data = convert(Matrix, iris[:,1:4])
data .-= mean(data, dims=1)
data ./= std(data, dims=1);

labels = convert(Vector, iris[:,end]);;

# -- define Turing model --
@model CRPMM(y) = begin

    α = 1.0
    D,N = size(y)

    rpm = DirichletProcess(α)

    z = tzeros(Int, N)
    m = tzeros(Float64, D, N)
    s = tzeros(Float64, D, N)

    for n in 1:N
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

        z[n] ~ ChineseRestaurantProcess(rpm, nk)
        if z[n] > K
            for d in 1:D
                s[d,z[n]] ~ InverseGamma(2, 3)
                m[d,z[n]] ~ Normal(0.0, sqrt(s[d,z[n]]))
            end
        end
        y[:,n] ~ MvNormal(m[:,z[n]], sqrt.(s[:,z[n]]))
    end
end

# -- sampling --
model = CRPMM(data')

nparticles = 100
niterations = 100

chain = sample(model, PG(nparticles, niterations));

# -- Extract assignments --
Z = Array(chain[:z]);

# Get number of clusters per iteration
k = [length(unique(Z[i,:])) for i in 1:size(Z,1)];

plot(k, ylabel = "Number of clusters", xlabel = "Iteration")
savefig("crp_clusters.png")

# extract true cluster assignments
y = [findfirst(l .== unique(labels)) for l in labels]

# compute rand index for each iteration
r = [randindex(y, Int.(Z[i,:]))[2] for i in 1:size(Z,1)]

plot(r, xlabel = "iteration", ylabel = "Rand index")
savefig("crp_randindex.png")