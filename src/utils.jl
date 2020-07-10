@static if VERSION < v"1.5.0-rc1.0"
    @nograd _cumsum(t::Dims) = Tuple(cumsum(collect(t)))
else
    _cumsum(t) = cumsum(t)
end

# TODO: decide on slice versus view
# Using view, there would be issues with CuArrays dispatch
function split_matrix(mat::AbstractMatrix, dims::Dims)
    map(_cumsum(dims), dims) do post, diff
        pre = post - diff + 1
        return mat[pre:post, :]
    end
end

# from Flux
glorot_uniform(dims...) = (rand(dims...) .- 0.5) .* sqrt(24 / sum(dims))
