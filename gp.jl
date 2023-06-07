#--------------------------------------------#
#=  Genetic Programming with Julia 
        by Nick Harris              =#
#--------------------------------------------#
using Base: String, Number, Float64, Bool, Symbol
using Random, Distributions
using StatsBase: countmap
using Match

include("elementary_functions.jl")

#--------------- Structs, Data Types, and Predefined Objects -------------------#

# Leaf Type, one node of tree
mutable struct Leaf
    value::Union{Int64, Float64, Nothing}
    type::Union{String, Nothing}
    single_op::Union{Int64, Nothing}
    left_child::Union{Leaf, Nothing}
    right_child::Union{Leaf, Nothing}
    num_children::Union{Int64, Nothing}
end

# Program Tree Type
struct Program_Tree
    depth::Int64
    root::Leaf
end

# Program_Tree population type - highest level object, to easily facilitate running an evolution experiment
mutable struct Tree_Pop
    pop::Array{Program_Tree}            # population of Program_Trees
    fitnesses::Array{Float64}           # fitnesses associated with each Program_Tree
    elitism::Int64                      # number of Program_Trees to preserve as-is to the next generation
    mutant_elitism::Int64               # number of Program_Trees that are mutant copies of those in the elitism pool,
                                            # added to the population in advance of crossover
    diversity_elitism::Int64            # number of Program_Trees with unique fitnesses to preserve as-is to the next generation
    mutant_diversity_elitism::Int64     # number of trees to add as mutant copies from the diversity_elitism pool,
                                          #  added to the population in advance of crossover
    diversity_generate::Int64           # number of randomly-initialized Program_Trees to add to the population every generation
    variable_preserve::Int64
    fitness_sharing::Bool               # When true, makes Program_Trees with the same fitness split the fitness amongst themselves 
                                            # stops one chromomsome from taking over the population; 
                                            # in evolutionary terms, stops niche-crowding
    selection_algorithm::String         # String specifying which selection algorithm (roulette, tournament, ranked, etc) to use in GA
    mutation_rate::Float64              # Float (0 to 1) specifying chance of a child chromosome being mutated
    max_tree_depth::Int64               # max depth of tree
    num_inputs::Int64                   # number of possible inputs for trees
    k_value                      # optional keyword argument, k-value for tournament selection
end

#--------------------------- Constructors ---------------------------------- #
#empty constructor for Leaf
function Leaf()
    return Leaf(nothing, nothing, nothing, nothing, nothing, 0)
end

#constructor for Program_Tree type
function Program_Tree(tree_depth, num_inputs)
    root = Leaf()
    initialize_tree_topology(tree_depth, 1, root)
    insert_children_count(root)
    initialize_tree_values(root, num_inputs)

    return Program_Tree(tree_depth, root)
end

#default constructor for Program_Tree population object
function Tree_Pop(pop_size::Int64, elitism::Int64, mutant_elitism::Int64, diversity_elitism::Int64, mutant_diversity_elitism::Int64, diversity_generate::Int64, variable_preseve::Int64,
     fitness_sharing::Bool, selection_algorithm::String, mutation_rate::Float64, max_tree_depth::Int64, num_inputs::Int64; k = 2)

    my_pop = [Program_Tree(max_tree_depth, num_inputs) for i ∈ 1:pop_size]
    fitnesses = collect(0.0 for i ∈ 1:pop_size)
    return Tree_Pop(my_pop, fitnesses, elitism, mutant_elitism, diversity_elitism, mutant_diversity_elitism, diversity_generate, variable_preseve, fitness_sharing, selection_algorithm, mutation_rate, max_tree_depth, num_inputs, k)
end

# ------------------------------ Tree Functions ----------------------------------------------#
# fill outs tree connections according to depth constraints
function initialize_tree_topology(tree_depth::Int64, depth::Int64, node::Leaf)
    if depth < tree_depth
        if rand() < 0.75
            node.left_child = Leaf()
            initialize_tree_topology(tree_depth, depth + 1, node.left_child)

            node.right_child = Leaf()
            initialize_tree_topology(tree_depth, depth + 1, node.right_child)
        end
    end
end

# fills out values of tree which already has its connections defined
function initialize_tree_values(node::Leaf, num_inputs::Int64)
    if node.left_child !== nothing && node.right_child !== nothing
        initialize_childed(node)
        initialize_tree_values(node.left_child, num_inputs)
        initialize_tree_values(node.right_child, num_inputs)
    else
        initialize_childless(node, num_inputs)
    end
end

# this is used to fill children counts 
function get_elements(root::Union{Leaf, Nothing})
    if root === nothing
        return 0
    end
    return (get_elements(root.left_child) + get_elements(root.right_child) + 1)
end

# inserts children count for each node
function insert_children_count(root::Union{Leaf, Nothing})
    if root === nothing
        return
    end
    root.num_children = get_elements(root) - 1
    insert_children_count(root.left_child)
    insert_children_count(root.right_child)
end

# returns number of children for root
function num_children(root::Union{Leaf, Nothing})
    if root === nothing
        return 0
    end
    return root.num_children + 1  
                                        
end

# helper function to return a random node
function random_node_util(root::Union{Leaf, Nothing}, count::Int64)
    if root === nothing
        return 0
    end

    if count == num_children(root.left_child)
        return root
    end

    if count < num_children(root.left_child)
        return random_node_util(root.left_child, count)
    end

    return random_node_util(root.right_child, count - num_children(root.left_child) - 1)
end

# returns a random node from a tree
function random_node(root::Leaf)
    count = rand(0:root.num_children)
    return random_node_util(root, count)
end

# prints a leaf in a formatted way
function print_leaf(p::Leaf)
    if p.type == "var"
        if p.single_op !== nothing
            print("<< $(sops_dict[p.single_op])(var$(p.value)) >>\n")
        else
            print("<< var$(p.value) >>\n")
        end
    elseif p.type == "const"
        if p.single_op !== nothing
            print("<< $(sops_dict[p.single_op])($(p.value)) >>\n")
        else
            print("<< $(p.value) >>\n")
        end
    elseif p.type == "op"
        if p.single_op !== nothing
            print("<< $(sops_dict[p.single_op])($(ops_dict[p.value])) >>\n")
        else
            print("<< $(ops_dict[p.value]) >>\n")
        end
    end
end

# prints a tree in a highly readable way
function print_tree(root::Leaf)
    if root === nothing
        return
    end
    print_leaf(root)
    print_subtree(root, "")
    print("\n")
end

# assists in printing tree in readable way
function print_subtree(root::Leaf, prefix::String)
    if root === nothing
        return
    end
    hasLeft = root.left_child !== nothing
    hasRight = root.right_child !== nothing
    if hasLeft == false && hasRight == false
        return
    end

    print(prefix)
    if hasLeft == false && hasRight == false
        return
    elseif hasLeft && hasRight
        print("├── ")
    elseif (hasLeft == false) && hasRight
        print("└── ")
    end

    if hasRight
        printStrand = hasLeft && hasRight && (root.right_child.right_child !== nothing || root.right_child.left_child !== nothing)
        newPrefix = ""
        if printStrand
            newPrefix = prefix * "│   "
        else
            newPrefix = prefix * "    "
        end
        print_leaf(root.right_child)
        print_subtree(root.right_child, newPrefix)
    end

    if hasLeft
        if hasRight
            print(prefix)
        end
        print("└── ")
        print_leaf(root.left_child)
        print_subtree(root.left_child, prefix * "    ")
    end
end

#calculates height of a tree
function height(leaf::Union{Leaf, Nothing})
    if leaf === nothing
        return 0
    end
    left_height = height(leaf.left_child)
    right_height = height(leaf.right_child)
    return max(left_height, right_height) + 1
end

# function to activate tree and compute result
function activate(node, inputs)
    @match node begin
        Leaf(value, "const", sop::Nothing, _, _, _) => return clamp(value, -100000, 100000)
        Leaf(value, "const", sop::Int64, _, _, _) => return clamp(single_op(sop, value), -100000, 100000)
        Leaf(value, "var", sop::Nothing, _, _, _) => return clamp(inputs[value], -100000, 100000)
        Leaf(value, "var", sop::Int64, _, _, _) => return clamp(single_op(sop, clamp(inputs[value], -100000, 100000)), -100000, 100000)
        Leaf(value, "op", sop::Nothing, left, right, _) => return clamp(operation(value, activate(left, inputs), activate(right, inputs)), -100000, 100000)
        Leaf(value, "op", sop::Int64, left, right, _) => return clamp(single_op(sop, operation(value, activate(left, inputs), activate(right, inputs))), -100000, 100000)
        _ => println("This is what I got, and I'm not happy about it: ($(node.value), $(node.type), $(node.single_op), $(node.left_child), $(node.right_child), $(node.num_children))")
    end
end

#second activation function to accept entire matrix and index for access (for working with shared arrays)
function activate(node, inputs, row_index)
    @match node begin
        Leaf(value, "const", sop::Nothing, _, _, _) => return clamp(value, -100000, 100000)
        Leaf(value, "const", sop::Int64, _, _, _) => return clamp(single_op(sop, value), -100000, 100000)
        Leaf(value, "var", sop::Nothing, _, _, _) => return clamp(inputs[row_index, value], -100000, 100000)
        Leaf(value, "var", sop::Int64, _, _, _) => return clamp(single_op(sop, clamp(inputs[row_index, value], -1000000, 100000)), -1000000, 100000)
        Leaf(value, "op", sop::Nothing, left, right, _) => return clamp(operation(value, activate(left, inputs, row_index), activate(right, inputs, row_index)), -1000000, 100000)
        Leaf(value, "op", sop::Int64, left, right, _) => return clamp(single_op(sop, operation(value, activate(left, inputs, row_index), activate(right, inputs, row_index))), -1000000, 100000)
        _ => println("This is what I got, and I'm not happy about it: ($(node.value), $(node.type), $(node.single_op), $(node.left_child), $(node.right_child), $(node.num_children))")
    end
end

# intializes a childless leaf
function initialize_childless(node::Leaf, num_inputs::Int64)
    if rand() < 0.66  #set node to a var
        node.type = "var"
        node.value = rand(1:num_inputs)
    else
        roll = rand()
        node.type = "const"
        if roll < 0.25 # set not to const in range (0, 1)
            node.value = rand() 
        elseif roll < 0.5  #set node to a predefined const
            node.value = consts_dict[rand(1:length(consts_dict))]
        elseif roll < 0.75    #set node to a random int in tight range
            node.value = rand(-12:1:12)
        else  #set node to a random float in wide range
            node.value = rand(-100.0:0.0001:100)
        end
    end

    if rand() < 0.33
        node.single_op = rand(1:length(sops_dict))
    end
end

# initializes a leaf with children (so an operation node)
function initialize_childed(node::Leaf)
    node.value = rand(1:length(ops_dict))
    node.type = "op"
    if rand() < 0.25
        node.single_op = rand(1:length(sops_dict))
    end
end

#prints program tree to a file so it can be saved and recovered later
function save_tree(node::Union{Leaf, Nothing}, outfile::IOStream)
    if node === nothing
        write(outfile, "∅ ")
    else
        sop_string = "⦱"
        if node.single_op !== nothing
            sop_string = string(node.single_op)
        end
        write(outfile, sop_string * "|" * node.type * "|" * string(node.value) * " ")
        save_tree(node.left_child, outfile)
        save_tree(node.right_child, outfile)
    end
end

#reads a tree from a file and loads it into object
function load_tree!(node::Leaf, infile::IOStream)
    s_list =  split(read(infile, String), " ")
    if length(s_list) < 1
        return
    end
    load_helper!(node, s_list)
end

#helper function to load tree from pre-processed string format
function load_helper!(node, s_list::Array)
    if length(s_list) < 1   #no more data to read
        return
    elseif s_list[1] == "∅" #node was recorded as empty
        popfirst!(s_list)
        return
    else # we should have a valid node on our hands
        element_list = split(s_list[1], "|")
        if element_list[1] != "⦱" #sop is not none 
            node.single_op = parse(Int32, element_list[1])
        end
        node.type = element_list[2]
        if node.type == "var" || node.type == "op"
            node.value = parse(Int64, element_list[3])
        else
            node.value = parse(Float64, element_list[3])
        end
        node.left_child = Leaf()
        node.right_child = Leaf()
        popfirst!(s_list)
        load_helper!(node.left_child, s_list)
        load_helper!(node.right_child, s_list)
        if node.type != "op"
            node.left_child = nothing
            node.right_child = nothing
        end
    end
end

#------------------------------ Population Functions ------------------------------------#

# function to compute the next population of the genetic algorithm 
#   Using selection, mutation, and crossover operations
function next_generation!(old_pop::Tree_Pop)
    new_pop = []
    new_fitnesses = []

    # sort chromosome-fitness pairs in order of fitness
    sorted_pairs = [[old_pop.pop[i] old_pop.fitnesses[i]] for i ∈ eachindex(old_pop.pop)]
    sort!(sorted_pairs, by=x->x[2], rev=true) #now in descending order
    sorted_fitnesses = [round(sorted_pairs[i][2], digits = 12) for i ∈ eachindex(old_pop.pop)]
    counter = countmap(sorted_fitnesses, alg= :dict)

    # preserve chromosomes according to elitism
    for i ∈ 1:old_pop.elitism
        push!(new_pop, sorted_pairs[i][1])
        push!(new_fitnesses, sorted_pairs[i][2])
    end

    # preserve chromosomes according to diversity_elitism
    diversity_count = 0
    pop_index = 1
    diversity_indices = []
    while diversity_count < old_pop.diversity_elitism && pop_index < length(old_pop.pop)
        push!(new_pop, sorted_pairs[pop_index][1])
        push!(new_fitnesses, sorted_pairs[pop_index][2])
        push!(diversity_indices, pop_index)
        diversity_count += 1
        pop_index += counter[sorted_fitnesses[pop_index]]
    end

    # add chromosomes according to mutant elitism
    for i ∈ 1:old_pop.mutant_elitism
        new_indiv = mutate(sorted_pairs[1][1], old_pop.num_inputs, old_pop.max_tree_depth)
        stop_cond = false
        while stop_cond == false
            if rand() < 0.5
                new_indiv = mutate(new_indiv, old_pop.num_inputs, old_pop.max_tree_depth)
            else
                stop_cond = true
                push!(new_pop, new_indiv)   
                push!(new_fitnesses, 0.0)           
            end
        end
    end

    #add chroms from mutant-diversity elitism
    mutant_diversity_count = old_pop.mutant_diversity_elitism
    if mutant_diversity_count > 0
        d_index = rand(diversity_indices)
        new_indiv = mutate(sorted_pairs[d_index][1], old_pop.num_inputs, old_pop.max_tree_depth)
        stop_cond = false
        while stop_cond == false
            if rand() < 0.5
                new_indiv = mutate(new_indiv, old_pop.num_inputs, old_pop.max_tree_depth)
            else
                stop_cond = true
                push!(new_pop, new_indiv)  
                push!(new_fitnesses, 0.0)
                mutant_diversity_count -= 1            
            end
        end
    end 

    # add new random individuals according to diversity_generate
    for i ∈ 1:old_pop.diversity_generate
        if length(new_pop) < length(old_pop.pop)
            push!(new_pop, Program_Tree(old_pop.max_tree_depth, old_pop.num_inputs))
            push!(new_fitnesses, 0.0)
        end
    end

    #add new individuals according to variable_preserve
    for i ∈ 1:old_pop.variable_preserve
        new_leaf = Leaf(i, "var", nothing, nothing, nothing, 0)
        new_tree = Program_Tree(1, new_leaf)
        push!(new_pop, new_tree)
        push!(new_fitnesses, 0.0)
    end

    normalized_fitnesses = []
    # adjust fitnesses according to fitness_sharing (if enabled)   #----------------- ALERT Pretty sure this fucks stuff up for some selection algorithms ------------------#
    if old_pop.fitness_sharing
        for i ∈ eachindex(sorted_fitnesses)
            if sorted_fitnesses[i] >= 0
                push!(normalized_fitnesses, Float64(sorted_fitnesses[i]/counter[sorted_fitnesses[i]]))
            else
                push!(normalized_fitnesses, Float64(sorted_fitnesses[i]*counter[sorted_fitnesses[i]]))
            end
        end
        #sort!(normalized_fitnesses, rev=true) #re-sorted after normalization  #this fucks shit up, doesn't it?
    else
        normalized_fitnesses = sorted_fitnesses
    end

    # perform selection algorithm to select parents for next generation
    if old_pop.selection_algorithm == "roulette"
         selections = roulette_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop)) 
    elseif old_pop.selection_algorithm == "tournament"
         selections = tournament_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop),  old_pop.k_value)
    elseif old_pop.selection_algorithm == "ranked"
         selections = ranked_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    elseif old_pop.selection_algorithm == "diversity_search"
         selections = diversity_search(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop), counter)
    elseif old_pop.selection_algorithm == "SUS"
         selections = stochastic_universal_sampling(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    else
        println("GP WARNING | '$(old_pop.selection_algorithm)' not a known selection algorithm. Defaulting to tournament-selection.")
         selections = tournament_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop),  old_pop.k_value)
    end
    
    # perform crossover between parents to create children, mutate with small probability, and populate next generation
    s_index = 1
    while length(new_pop) < length(old_pop.pop)
        #crossover to create child
        if rand() < 0.9  #90% chance to perform crossover, but small chance a chromosome is passed uncrossed 
            child = crossover(sorted_pairs[selections[s_index][1]][1], sorted_pairs[selections[s_index][2]][1], old_pop.max_tree_depth, old_pop.num_inputs)
        else
            choice = rand(1:2)
            child = sorted_pairs[selections[s_index][choice]][1]
        end
        
        #mutate child with some probability
        if rand() < old_pop.mutation_rate
            child = mutate(child, old_pop.num_inputs, old_pop.max_tree_depth)
        end
        push!(new_pop, child)    
        push!(new_fitnesses, 0.0)
        s_index += 1
    end

    old_pop.pop = new_pop
    old_pop.fitnesses = new_fitnesses
end

# performs fitness-proportional (aka roulette-wheel) selection and returns indices of selected parents
function roulette_selection(fitnesses, num_selections) 
    selections = []

    #build roulette wheel (from ratios of fitnesses to fitness total)
    roulette_wheel = []
    fitness_proportions = []
    fitness_total = sum(fitnesses)
    for i ∈ eachindex(fitnesses)
        push!(fitness_proportions, fitnesses[i]/fitness_total)
        i == 1 ? push!(roulette_wheel, fitness_proportions[i]) : push!(roulette_wheel, roulette_wheel[i - 1] + fitness_proportions[i])
    end

    points = sort([rand() for i ∈ 1:(2*num_selections)])
    flat_selections = []

     j = 1 
    for i ∈ 1:(2*num_selections)       
        while roulette_wheel[j] < points[i]
             j
            j += 1
        end
        push!(flat_selections, j)
    end
    
    for i ∈ 1:num_selections
        push!(selections, [flat_selections[(2*i) - 1] flat_selections[2*i]])
    end

    return selections
end

# performs stochastic univeral sampling selection and returns indices of selected parents
function stochastic_universal_sampling(fitnesses, num_selections)
    selections = []

    #build roulette wheel (from ratios of fitnesses to fitness total)
    roulette_wheel = []
    fitness_proportions = []
    fitness_total = sum(fitnesses)
    for i ∈ eachindex(fitnesses)
        push!(fitness_proportions, fitnesses[i]/fitness_total)
        i == 1 ? push!(roulette_wheel, fitness_proportions[i]) : push!(roulette_wheel, roulette_wheel[i - 1] + fitness_proportions[i])
    end

    N = 2*num_selections
    p_distance = 1.0/N
    start = rand()*p_distance
    points = [start + i*p_distance for i ∈ 0:(N-1)]

    flat_selections = []

     j = 1 
    for i ∈ 1:N       
        while roulette_wheel[j] < points[i]
             j
            j += 1
        end
        push!(flat_selections, j)
    end
    
    for i ∈ 1:num_selections
        push!(selections, [flat_selections[(2*i) - 1] flat_selections[2*i]])
    end

    return selections
end

# performs diversity-search selection and returns indices of selected parents
function diversity_search(fitnesses, num_selections, counter)
    selections = []

    eligible_individuals = []

    #adjust fitnesses (all unique fitnesses equal; all not-unique fitnesses = 0)
    for i ∈ eachindex(fitnesses)
        #if count(j->(abs(j - fitnesses[i]) < 0.0000000001), fitnesses) < 2
        if counter[fitnesses[i]] < 2
            push!(eligible_individuals, i)  
        end
    end

    for i ∈ 1:num_selections
        spin1 = rand(1:length(eligible_individuals)) 
        spin2 = rand(1:length(eligible_individuals))  
        push!(selections, [eligible_individuals[spin1] eligible_individuals[spin2]])
    end

    return selections
end

# performs k-way tournament selection and returns indices of selected parents
function tournament_selection(fitnesses, num_selections::Int64, k_value::Int64) 
    selections = []
    k = k_value

    for i ∈ 1:num_selections      
        candidates = collect(rand(1:length(fitnesses)) for j ∈ 1:k)
        cand_fitnesses = collect(fitnesses[c] for c ∈ candidates)    
        winner1 = candidates[argmax(cand_fitnesses)]

        candidates = collect(rand(1:length(fitnesses)) for j ∈ 1:k)
        cand_fitnesses = collect(fitnesses[c] for c ∈ candidates)
        winner2 = candidates[argmax(cand_fitnesses)]

        push!(selections, [winner1 winner2])
    end

    return selections
end

# version of tournament selection with even less selective pressure than k=2, taking a float value for k
function tournament_selection(fitnesses, num_selections::Int64, k_value::Float64) 
    selections = []
    k = 2

    for i ∈ 1:num_selections      
        candidates = collect(rand(1:length(fitnesses)) for j ∈ 1:k)
        cand_fitnesses = collect(fitnesses[c] for c ∈ candidates)  

        win_index = argmax(cand_fitnesses)
        
        if rand() < k_value
            winner1 = candidates[win_index]
        else
            if win_index == 1
                winner1 = candidates[2]
            else
                winner1 = candidates[1]
            end
        end

        candidates = collect(rand(1:length(fitnesses)) for j ∈ 1:k)
        cand_fitnesses = collect(fitnesses[c] for c ∈ candidates)
        
        win_index = argmax(cand_fitnesses)
        
        if rand() < k_value
            winner2 = candidates[win_index]
        else
            if win_index == 1
                winner2 = candidates[2]
            else
                winner2 = candidates[1]
            end
        end

        push!(selections, [winner1 winner2])
    end

    return selections
end

# performs ranked selection and returns indices of selected parents
function ranked_selection(fitnesses, num_selections) 
    ranked_fitnesses = reverse(collect(1:length(fitnesses)))
    return roulette_selection(ranked_fitnesses, num_selections)
end

#Function to trim tree down to size if it exceeds max_depth
function trim!(node::Leaf, max_depth, current_depth, num_inputs)
    if current_depth == max_depth
        node.left_child = nothing
        node.right_child = nothing

        if node.type == "op"
            node = initialize_childless(node, num_inputs)
        end
        return nothing
    end

    if node.left_child !== nothing
        trim!(node.left_child, max_depth, current_depth + 1, num_inputs)
        trim!(node.right_child, max_depth, current_depth + 1, num_inputs)
    end

    return nothing
end

# performs crossover operations between trees
function crossover(t1::Program_Tree, t2::Program_Tree, max_tree_depth::Int64, num_inputs)        
    roll = rand()
    #copy over parents into new objects
    if roll < 0.5
        chrom1 = deepcopy(t1)
        chrom2 = deepcopy(t2)
    else
        chrom1 = deepcopy(t2)
        chrom2 = deepcopy(t1)
    end

    #select nodes to cross (segments of trees to transplant)
    node1 = random_node(chrom1.root)
    node2 = random_node(chrom2.root)

    #cross nodes (only 50% chance to grab single-op as well)
    temp = Leaf()
    
    temp.type = node1.type
    temp.value = node1.value
    if rand() < 0.5 
        temp.single_op = node1.single_op
    end
    temp.left_child = node1.left_child
    temp.right_child = node1.right_child

    node1.type = node2.type
    node1.value = node2.value
    if rand() < 0.5
        node1.single_op = node2.single_op
    end
    node1.left_child = node2.left_child
    node1.right_child = node2.right_child

    node2.type = temp.type
    node2.value = temp.value
    node2.single_op = temp.single_op
    node2.left_child = temp.left_child
    node2.right_child = temp.right_child

    #return if resulting tree does not exceed height limit
    if height(chrom1.root) <= max_tree_depth
        insert_children_count(chrom1.root)
        return chrom1
    elseif height(chrom2.root) <= max_tree_depth
        insert_children_count(chrom2.root)
        return chrom2
    else
        if rand() < 0.5
            trim!(chrom1, max_tree_depth, 1, num_inputs)
            insert_children_count(chrom1.root)
            return chrom1
        else
            trim!(chrom2, max_tree_depth, 1, num_inputs)
            insert_children_count(chrom2.root)
            return chrom2
        end
    end
end

# mutates a program tree by altering one node at random
function mutate(t::Program_Tree, num_inputs::Int64, max_height::Int64)
    if rand() < 0.5 # strong mutate: completely replace node with random one  
        t2 = deepcopy(t)
        
        #select node to mutate
        new_node = random_node(t2.root)

        roll = rand()
        if roll < 0.33  #set node to an op
            initialize_childed(new_node)
            if new_node.left_child === nothing
                new_node.left_child = Leaf()
                initialize_childless(new_node.left_child, num_inputs)
            end
            if new_node.right_child === nothing
                new_node.right_child = Leaf()
                initialize_childless(new_node.right_child, num_inputs)
            end
        else  #set node to a var or const
            new_node.left_child = nothing
            new_node.right_child = nothing
            initialize_childless(new_node, num_inputs)
        end

        if height(t2.root) <= max_height
            insert_children_count(t2.root)
            return t2
        else
            trim!(t2.root, max_height, 1, num_inputs)
            insert_children_count(t2.root)
            return t2
        end
    else  #weak mutate, just modify one component of node
        t2 = deepcopy(t)
        new_node = random_node(t2.root)

        roll = rand()
        if roll < 0.25 #modify sop
            if rand() < 0.15
                new_node.single_op = nothing
            else
                new_node.single_op = rand(1:length(sops_dict))
            end
        else
            if new_node.type == "op"
                new_node.value = rand(1:length(ops_dict))
            elseif new_node.type == "var"
                new_node.value = rand(1:length(num_inputs))
            else
                # Center gaussian distribution on constant and modify it
                sig = abs(new_node.value)/10.0
                if sig == 0
                    sig = 0.01
                end
                dist = Normal( new_node.value, sig )
                new_node.value = rand(dist)  
            end
        end

        return t2
    end
end