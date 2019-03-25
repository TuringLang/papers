using HDF5, JLD

const nbmnistdata = load(splitdir(Base.@__DIR__)[1]*"/aistats2018/mnist-10000-40.data")["data"]