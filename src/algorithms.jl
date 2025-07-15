using LinearAlgebra
using KrylovKit


"""
    naive_arnoldi(k::Int)

Return a function which takes a Markov chain `P` with initial distribution
`p_0` and returns its Arnoldi aggregation of size `k`.

Made to collect data.
"""
function naive_arnoldi(k::Int)
     return (P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}) -> _naive_arnoldi_impl(P, p_0, k)
end

"""
    arnoldi_with_π(k::Int)

Return a function which takes a Markov chain `P` with initial distribution
`p_0` and returns its Arnoldi aggregation of size `k` with aggregated
stationary distribution `π_st`.

Made to collect data.
"""
function arnoldi_with_π(k::Int)
     return (P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}) -> _arnoldi_with_π_impl(P, p_0, k)
end

"""
    arnoldi_with_π_dynamic(ε::Float64)

Return a function which takes a Markov chain `P` with initial distribution
`p_0` and returns its smallest Arnoldi aggregation with ⟨|π|, |ΠA-AP|⋅1_n⟩ < ε.

Made for actual application.
"""
function arnoldi_with_π_dynamic(ε::Float64, measure_sizes::Array{Int})
     return (P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}) -> _arnoldi_with_π_dynamic_impl(P, p_0, ε, measure_sizes)
end

"""
    _naive_arnoldi_impl(P::AbstractMatrix, p_0::AbstractVector, k::Int,
                orth::KrylovKit.Orthogonalizer)

Compute the aggregated step matrix `Π` of size `k`, disaggregation matrix
`A` and aggregated initial distribution `π_0` by using the Arnoldi
aggregation for a Markov chain `P` and initial distribution `p_0`.
"""
function _naive_arnoldi_impl(P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}, k::Int)
    iterator = ArnoldiIterator(P, p_0, ModifiedGramSchmidt())
    factorization = initialize(iterator)
    # Starts with aggregation size 1, so k - 1 steps, instead of k
    [expand!(iterator, factorization) for _ in 1:k - 1]
    A = stack(basis(factorization))
    Π = rayleighquotient(factorization)
    π_0 = Float64[norm(p_0, 2); zeros(Float64, k - 1)]
    # Return arbitrary "eigenvector", as it is not computed with this method
    return Π, A, π_0, π_0
end

"""
    _arnoldi_with_π_impl(P::AbstractMatrix, p_0::AbstractVector, k::Int)

Compute the aggregated step matrix `Π` of size `k`, disaggregation matrix
`A`, aggregated initial distribution `π_0` and aggregated stationary
distribution `π_st` by using the Arnoldi aggregation for a Markov chain `P`
and initial distribution `p_0` along with Krylov-Schur for `π_st` in the end.
"""
function _arnoldi_with_π_impl(P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}, k::Int)
    iterator = ArnoldiIterator(P, p_0, ModifiedGramSchmidt())
    factorization = initialize(iterator)
    # Starts with aggregation size 1, so k-1 steps, instead of k
    [expand!(iterator, factorization) for _ in 1:k - 1]
    A = stack(basis(factorization))
    Π = rayleighquotient(factorization)
    π_0 = Float64[norm(p_0, 2); zeros(Float64, k - 1)]
    _, vecs, _ = eigsolve(Π)
    # Bad convergence has happened
    if imag(vecs[1][1]) != 0
        println("Bad convergence of eigenpair at aggregation size ", k)
        return Π, A, π_0, π_0
    end
    π_st = vecs[1]
    π_st .*= 1.0 / norm(A * π_st, 1)
    return Π, A, π_st, π_0
end

"""
    _arnoldi_with_π_dynamic_impl(P::AbstractMatrix, p_0::AbstractVector, ε::Float64, measure_sizes::Array{Int})

Compute the smallest aggregated step matrix `Π`, disaggregation matrix
`A`, aggregated initial distribution `π_0` and aggregated stationary
distribution `π_st` with ⟨|π|, |ΠA - AP|⋅1_n⟩ < ε by using the Arnoldi
aggregation for a Markov chain `P` and initial distribution `p_0` along
with Krylov-Schur for `π_st` in selected sizes `measure_sizes`. Further,
we bisect between the last and second to last size at which the criterion
was measured to find a smaller aggregation fulfilling said criterion if
`do_bisection` is set to true.

The case of `current_size` being so small that `π_st` has not yet converged
is not handled, as any usable aggregations needs to be bigger anyways.
"""
function _arnoldi_with_π_dynamic_impl(P::AbstractMatrix{Float64}, p_0::AbstractVector{Float64}, ε::Float64, measure_sizes::Array{Int})
    iterator = ArnoldiIterator(P, p_0, ModifiedGramSchmidt())
    factorization = initialize(iterator)
    current_size = 1
    original_size = length(p_0)
    max_size = 2000 # We usually do not have larger aggregations than that, if we do, increase this value
    # Allocate with dummy values for more efficient access later
    A = zeros(Float64, (original_size, max_size))
    Π = zeros(Float64, (max_size, max_size))
    π_st = zeros(ComplexF64, max_size)
    for size in measure_sizes
        while current_size < size
            expand!(iterator, factorization)
            current_size += 1
        end
        # Measure criterion as we reached the next measure size
        A[1:original_size,1:current_size] = stack(basis(factorization))
        Π[1:current_size, 1:current_size] = rayleighquotient(factorization)            
        _, vecs, _ = eigsolve(@view Π[1:current_size, 1:current_size])
        π_st[1:current_size] = vecs[1]
        π_st[1:current_size] .*= 1.0 / norm((@view A[:, 1:current_size]) * (@view π_st[1:current_size]), 1)
        criterion = dot(abs.((@view π_st[1:current_size])), ones(Float64, length(p_0))' * abs.((@view A[:, 1:current_size]) * (@view Π[1:current_size, 1:current_size]) - P * (@view A[:, 1:current_size])))
        # If the eigenvector is complex, it certainly has not converged (and we can't use complex vectors for probabilities)
        if criterion <= ε && imag((@view π_st[1:current_size])[1]) == 0.0
            break
        end
    end
    π_0 = Float64[norm(p_0, 2); zeros(Float64, current_size - 1)]
    return  Π[1:current_size, 1:current_size], A[1:original_size,1:current_size],  real(π_st[1:current_size]), π_0
end
