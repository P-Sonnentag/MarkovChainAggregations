using LinearAlgebra

abstract type Aggregation end

"""
    ArnoldiAggregation

Save all necessary information about an aggregation.

Includes all variables which uniquely define an aggregation.
Note that transient distributions are manipulated in-place,
so no "history" is available.

Built for speed and real-world application.

* `P`: Transition matrix
* `Π`: aggregated step matrix
* `π_st`: aggregated stationary distribution
* `π_k`: aggregated transient distribution
* `tmp_π_k`: preallocated temporary storage
* `A`: disaggregation matrix
"""
mutable struct ArnoldiAggregation <: Aggregation
    P::AbstractMatrix{Float64}
    Π::Matrix{Float64}
    π_st::Vector{Float64}
    π_k::Vector{Float64}
    tmp_π_k::Vector{Float64}
    A::Matrix{Float64}
end

"""
    ArnoldiAggregation(P::AbstractMatrix, p_0::AbstractVector, algo::Function)::ArnoldiAggregation

Constructor for the `ArnoldiAggregation` struct. Calculate the
aggregation with the given algorithm `algo` using the transition
matrix `P` and initial distribution `p_0`.
"""
function ArnoldiAggregation(P::AbstractMatrix{Float64}, p_0::Vector{Float64}, algo::Function)::ArnoldiAggregation
    Π, A, π_st, π_0 = algo(P, p_0)
    return ArnoldiAggregation(P, Π, π_st, π_0, copy(π_0), A)
end

"""
    step!(aggregation::ArnoldiAggregation)

Step to the next aggregated transient distribution.

Uses optimized in-place BLAS-operations internally.
"""
function step!(aggregation::ArnoldiAggregation)
    #println(size(aggregation.tmp_π_k))
    mul!(aggregation.tmp_π_k, aggregation.Π, aggregation.π_k)
    # Rearrange pointers for consistency in further steps
    aggregation.π_k, aggregation.tmp_π_k = aggregation.tmp_π_k, aggregation.π_k
end

"""
    ArnoldiAggregationData

Saves additional information about an Arnoldi aggregation, like errors.

Made to collect data and observe behaviors of Arnoldi aggregations; NOT for speed.

* `base`: bare bones Arnoldi aggregation
* `p_k`: transient distribution
* `tmp_p_k`: preallocated temporary storage
* `p̃_st`: approximated stationary distribution
* `p̃_k`: approximated transient distribution
* `tmp_p̃_k`: preallocated temporary storage
* `err`: static error
* `err_k`: dynamic error
* `err_k_bnd`: dynamic error bound
* `err_st`: error of `p̃_st` not being invariant under `P`
* `err_π_st`: ⟨|π|, |ΠA-AP|⋅1_n⟩
* `diff`: |ΠA-AP|
"""
mutable struct ArnoldiAggregationData <: Aggregation
    base::ArnoldiAggregation
    p_k::Vector{Float64}
    tmp_p_k::Vector{Float64}
    p̃_st::Vector{Float64}
    p̃_k::Vector{Float64}
    tmp_p̃_k::Vector{Float64}
    err::Float64
    err_k::Float64
    err_k_bnd::Float64
    err_st::Float64
    err_π_st::Float64
    diff::Matrix{Float64}
end

"""
    ArnoldiAggregationData(P::AbstractMatrix, p_0::AbstractVector, algo::Function)::ArnoldiAggregation

Constructor for the `ArnoldiAggregationData` struct. Calls the base
constructor to build the actual aggregation, while also initiating
further values/variables of interest.
Further calculates many other important values for evaluation.
"""
function ArnoldiAggregationData(P::AbstractMatrix{Float64}, p_0::Vector{Float64}, algo::Function)::ArnoldiAggregationData
    base = ArnoldiAggregation(P, p_0, algo)
    p̃_st = A * π_st
    diff = abs.(A * Π - P * A)
    err = opnorm(diff, 1)
    err_st = norm(p̃_st - P * p̃_st, 1)
    err_π_st = dot(abs.(π_st), ones(Float64, size(P)[1])' * diff)
    return ArnoldiAggregationData(base, p_0, copy(p_0), p̃_st, copy(p_0), copy(π_0), err, 0.0, 0.0, err_st, err_π_st, diff)
end

"""
    step!(aggregation::ArnoldiAggregationData)

Step to the next aggregated transient distribution by using the base step!-function.

Uses optimized in-place BLAS-operations internally.
"""
function step!(aggregation::ArnoldiAggregationData)
    step!(aggregation.base)
end

"""
    step_all!(aggregation::ArnoldiAggregationData)

Step to the next transient and aggregated transient distribution and
compute current dynamic error bound.

Uses optimized in-place BLAS-operations internally.
"""
function step_all!(aggregation::ArnoldiAggregationData)
    mul!(aggregation.base.tmp_π_k, aggregation.base.Π, aggregation.base.π_k)
    mul!(aggregation.tmp_p_k, aggregation.base.P, aggregation.p_k)
    # Rearrange pointers for consistency in further steps
    aggregation.base.π_k, aggregation.base.tmp_π_k, aggregation.tmp_p̃_k = aggregation.base.tmp_π_k, aggregation.base.π_k, aggregation.base.tmp_π_k
    aggregation.p_k, aggregation.tmp_p_k = aggregation.tmp_p_k, aggregation.p_k
    
    aggregation.err_k_bnd += dot(abs.(aggregation.base.π_k), ones(Float64, size(aggregation.base.P)[1])' * aggregation.diff)
end

"""
    getDynamicError!(aggregation::ArnoldiAggregationData, norm_mode::Int)

Calculate the current dynamic error and then call `step_all!()`.
"""
function getDynamicError!(aggregation::ArnoldiAggregationData)
    mul!(aggregation.p̃_k, aggregation.base.A, aggregation.tmp_p̃_k)
    aggregation.err_k = _l1norm_diff(aggregation.p̃_k, aggregation.p_k)
    step_all!(aggregation)
end

function _l1norm_diff(a::Vector{Float64}, b::Vector{Float64})::Float64
    s::Float64 = 0.0
    @inbounds @simd for i in eachindex(a, b)
        s += abs(a[i] - b[i])
    end
    return s
end
