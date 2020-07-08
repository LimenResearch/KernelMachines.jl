@static if VERSION < v"1.5.0-rc1.0"
    @nograd _cumsum(t::Dims) = Tuple(cumsum(collect(t)))
else
    _cumsum(t) = cumsum(t)
end

# TODO: slice versus view, issue with CuArrays dispatch
function slice(mat::AbstractMatrix, list::Dims)
    map(_cumsum(list), list) do post, diff
        pre = post - diff + 1
        return mat[pre:post, :]
    end
end

# from Flux
glorot_uniform(dims...) = (rand(dims...) .- 0.5) .* sqrt(24 / sum(dims))
