include("algorithms.jl")
include("aggregation.jl")
include("io.jl")
using BenchmarkTools

function arnoldi_experiment()
    P = load_transition_matrix("src/rsvp_m7n5mn3_uniformized.tra")
    p_0 = rand(Float64, 842)
    normalize!(p_0, 1)

    measure_sizes = [1 + 10 * i for i in 0:200]
    ε = 1e-12

    @btime begin
        algo = arnoldi_with_π_dynamic($ε, $measure_sizes)
        aggregation = ArnoldiAggregation($P, $p_0, algo)
        #for _ in 1:1e5
        #    step!(aggregation)
        #end
    end

    algo = arnoldi_with_π_dynamic(ε, measure_sizes)
    aggregation = ArnoldiAggregation(P, p_0, algo)
    println(size(aggregation.π_st)[1])
end
arnoldi_experiment()
