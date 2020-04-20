# Loading

using MLDatasets

image_raw = MNIST.convert2features(MNIST.traintensor(Float64))
label = MNIST.trainlabels() .+ 1

# Pre-processing

using MultivariateStats

D_pca = 40
pca = fit(PCA, image_raw; maxoutdim=D_pca)

image = transform(pca, image_raw)

@info "Peformed PCA to reduce the dimension to $D_pca"

# Data function

get_data(n=1_000) = Dict(
    "C" => 10,
    "D" => D_pca,
    "N" => n,
    "image" => copy(image[:,1:n]'),
    "label" => label[1:n],
)
