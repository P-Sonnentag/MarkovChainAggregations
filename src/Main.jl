include("algorithms.jl")
include("aggregation.jl")
include("io.jl")

function arnoldi_experiment()
    P = load_transition_matrix("src/rsvp_m7n5mn3_uniformized.tra")
    p_0 = rand(Float64, 842)
    normalize!(p_0, 1)

    measure_sizes = [1 + 10 * i for i in 0:100]
    ε = 1e-12

    algo = arnoldi_with_π_dynamic(ε, measure_sizes)
    aggregation = ArnoldiAggregation(P, p_0, algo)
    for _ in 1:1e5
        step!(aggregation)
    end

    algo = arnoldi_with_π_dynamic(ε, measure_sizes)
    aggregation = ArnoldiAggregation(P, p_0, algo)
end
arnoldi_experiment()
