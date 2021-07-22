# test to find program tree that correctly computes the area of a circle
#   or volume of a sphere
# Also testing various aspects of julia GP and parallelism
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

    return max(volume_sum - error, 0)
end

@everywhere function evaluate_batch(chroms::Array{Program_Tree})
    fs= []
    for tree ∈ chroms
        push!(fs, evaluate_chrom(tree))
    end

    return fs
end

println("global funcs concluded")

#--------------- Evolution Experiment ------------------ #
println("\n\nStarting Evolution Test\n")

#Random.seed!(1234)

println("pop init: ") 

#parameters: pop_size, elitism, diversity_elitism, diversity_generate, fitness_sharing, selection_algorithm, mutation_rate, max_tree_depth, num_inputs
@time global my_pop = Tree_Pop(42000, 100, 0, 4200, false, "tournament", 0.20, 4, 1)
global MAX_GENS = 1000

global stop_cond = false
global gen_count = 1
global lowest_error = 999999999999.999

while stop_cond == false
    global gen_count
    global lowest_error
    global stop_cond
    global my_pop
    global volume_sum
    global MAX_GENS

    println("  GENERATION $gen_count   ")
     
    p_fitnesses = []
    fair_share = round(Int, length(my_pop.pop)/(num_procs - 1))

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

    max_fitness = my_pop.fitnesses[argmax(my_pop.fitnesses)]
    best_tree = my_pop.pop[argmax(my_pop.fitnesses)]
    best_error = volume_sum - max_fitness

    if best_error < (lowest_error - 0.001)
        lowest_error = best_error
        print("\n")
        println("   best_error: $best_error ($((best_error/volume_sum)*100)% error of true result)\n")
        print_tree(best_tree.root)
        if best_error < 0.0000000001
            break
        end
    end

    @time next_generation!(my_pop)

    gen_count += 1
    gen_count > MAX_GENS ? stop_cond = true : "egg"

    if gen_count > MAX_GENS
        io = open("saved_trees.txt", "w")
        save_tree(best_tree.root, io)
        close(io)   
        io = open("saved_trees.txt", "r")
        loaded_tree = Program_Tree(1, Leaf())
        load_tree!(loaded_tree.root, io)
        close(io)

        loaded_error = volume_sum - evaluate_chrom(loaded_tree)
        println("\n error of loaded tree: $loaded_error\n")

        print_tree(loaded_tree.root)

        io2 = open("saved_loaded.txt", "w")
        save_tree(loaded_tree.root, io2)
        close(io2)
    end
end