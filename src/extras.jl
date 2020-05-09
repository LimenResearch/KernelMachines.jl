struct Linear{T}
    W::T
end

Linear(a, b) = Linear(glorot_uniform(b, a))
Linear((a, b)::Tuple) = Linear(a, b)

@functor Linear

(l::Linear)(v) = l.W * v

dot(l1::Linear, l2::Linear) = dot(l1.W, l2.W)

struct Splitter{L<:Tuple}
    layers::L
end

Splitter(args...) = Splitter(args)

@functor Splitter

(s::Splitter)(t) = map(l -> l(t), s.layers)

dot(s1::Splitter, s2::Splitter) = sum(map(dot, s1.layers, s2.layers))