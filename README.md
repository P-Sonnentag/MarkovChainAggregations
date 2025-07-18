# MarkovChainAggregations

This is currently just a small script to build Arnoldi aggregations of DTMCs. The Markov chain must be given as a sparse matrix or as a .tra file (see rsvp_m7n5mn3_uniformized.tra as an example of the format). The inital distribution is just given as a vector.

## Installation

This software uses the standard Julia package LinearAlgebra as well as KrylovKit (see https://github.com/Jutho/KrylovKit.jl). KrylovKit must be installed through Pkg.

## License

This software is licensed under the MIT license.
