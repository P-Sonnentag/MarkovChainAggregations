using SparseArrays

function load_transition_matrix(filename::String)
    open(filename, "r") do file
        first_line = readline(file)
        _, num_transitions = split(first_line)
        num_transitions = parse(Int, num_transitions)

        rows = Int[]
        cols = Int[]
        values = Float64[]
        
        for _ in 1:num_transitions
            line = readline(file)
            col, row, val = split(line)
            push!(rows, parse(Int, row))
            push!(cols, parse(Int, col))
            push!(values, parse(Float64, val))
        end

        # .tra-files start at zero, julia at one, so we shift
        rows .+= 1
        cols .+= 1

        # Zero-transitions don't matter in Markov chains
        return dropzeros(sparse(rows, cols, values))
    end
end

function save_data_to_file(full_filepath::String,
                    datapoints::Array)    
    open(full_filepath, "a") do file
        println(file, join(string.(datapoints), " "))
    end
end
