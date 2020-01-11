using DelimitedFiles

function get_data(t=2610)
    data_raw, header = readdlm(joinpath(@__DIR__, "data.csv"), ',', header=true)
    y = map(x -> isa(x, Number) ? x : 0.1, data_raw[:,2])[1:t]
    return Dict(
        "T" => length(y),
        "y" => y
    )
end
