# test to find program tree that correctly computes the area of a circle
#   or volume of a sphere
#     Also testing various aspects of julia GP and parallelism
using Random: Float64, length
using Distributed



println("--------------")
println(length(Sys.cpu_info()))
addprocs(7)
num_procs = nprocs() 
println(num_procs)
println("--------------")

@everywhere include("gp.jl")
println("include concluded")

global areas = [π*(r^2) for r ∈ 1:20]
@everywhere global volumes = [(4.0/3.0)*π*(r^3) for r ∈ 1:20]
@everywhere volume_sum = sum(volumes)

println("global vars concluded")

@everywhere function evaluate_chrom(t::Program_Tree)
    error = 0.0
    global R = 1
    for i ∈ 1:length(volumes)
        error += abs(activate(t.root, [R]) - volumes[i])
        R += 1
    end

    return volume_sum - error
end

@everywhere function evaluate_batch(chroms::Array{Program_Tree})
    fs= []
    for tree ∈ chroms
        push!(fs, evaluate_chrom(tree))
    end

    return fs
end

println("global funcs concluded")

#--------------- Testing same problem with evolution ------------------ #
println("\n\nStarting Evolution Test\n")

Random.seed!(1234)

println("pop init: ") 
#parameters: pop_size, elitism, diversity_elitism, diversity_generate, fitness_sharing, selection_algorithm, mutation_rate, max_tree_depth, num_inputs
@time global my_pop = Tree_Pop(21000, 5000, 0, 500, false, "tournament", 0.20, 4, 1)

global stop_cond = false
global gen_count = 1
global lowest_error = 999999999999.999


while stop_cond == false
    global gen_count
    global lowest_error
    global stop_cond
    global my_pop
    global volume_sum

    println("  GENERATION $gen_count   ")
    
    
    p_fitnesses = []
    fair_share = Int64(floor(length(my_pop.pop)/num_procs))

    # batch parallel  
    batch_index = 1    
    @time begin
        for i ∈ 1:(num_procs - 1)
            if i < (num_procs - 1)
                push!(p_fitnesses, remotecall(evaluate_batch, i + 1, my_pop.pop[batch_index:(batch_index + fair_share - 1)]))
                batch_index += fair_share
            else
                push!(p_fitnesses, remotecall(evaluate_batch, i + 1, my_pop.pop[batch_index:length(my_pop.pop)]))
            end
        end

        f_fitnesses = []
        for i ∈ 1:(num_procs - 1)
            append!(f_fitnesses, fetch(p_fitnesses[i]))
        end
        for i ∈ 1:length(my_pop.pop)
            my_pop.fitnesses[i] = f_fitnesses[i]
        end  
    end

    #non-parallel 
    #=
    @time for i ∈ 1:length(my_pop.pop)
        my_pop.fitnesses[i] = evaluate_chrom(my_pop.pop[i])
    end    
    =#

    max_fitness = my_pop.fitnesses[argmax(my_pop.fitnesses)]
    best_tree = my_pop.pop[argmax(my_pop.fitnesses)]
    best_error = volume_sum - max_fitness

    if best_error < (lowest_error - 0.001)
        lowest_error = best_error
        println("   best_error: $best_error") #, best tree: $(best_tree)")
    end

    @time j next_generation!(my_pop)

    gen_count += 1
    gen_count > 1000 ? stop_cond = true : "egg"
end