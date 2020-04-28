struct Linear{T}
    W::T
end

Linear(a, b) = Linear(glorot_uniform(b, a))
Linear((a, b)::Tuple) = Linear(a, b)

@functor Linear

(l::Linear)(v) = l.W * v

dot(l1::Linear, l2::Linear) = dot(l1.W, l2.W)