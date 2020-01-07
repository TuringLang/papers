using FillArrays

@model high_dim_gauss(D) = begin
    m ~ MvNormal(Fill(0, D), 1)
end

get_model(args...) = high_dim_gauss(args...)