using DelimitedFiles

function get_data()
    data_raw, header = readdlm("data.csv", ',', header=true)
    y = map(x -> isa(x, Number) ? x : 0.1, data_raw[:,2])
    return Dict(
        "T" => length(y),
        "y" => y
    )
end
