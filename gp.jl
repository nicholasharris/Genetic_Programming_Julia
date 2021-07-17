#--------------------------------------------#
#=  Genetic Programming with julia 
        by Nick Harris              =#
#--------------------------------------------#
using Base: String, Number, Int64, Bool, Symbol
using Random

# ------------- Elementary functions for use in program trees -------------- #
function mean(a, b)
    return (a + b)/2.0
end

function safe_divide(a, b)
    if (abs(b) < 0.001)
        return 1
    else
        return a/b
    end 
end

function safe_modulo(a, b)
    if (b > 0)
        return a % ceil(b)
    else
        return 1.0
    end
end
function square(x)
    return x^2.0;
end

function cube(x)
    return x^3.0;
end

function negate(x)
    return -1.0 * x;
end

function inverse(x)
    result = 0
    abs(x) > 0.001 ? result = 1.0/x : result = 1.0
    return result
end

function ln(x)
    result = 0
    x > 0.001 ? result = log(x) : result = 1.0
    return result
end

function psqrt(x)
    return √(abs(x))
end

function sigmoid(x)
    return 1.0/(1 + ℯ^(-x))
end

#gompertz is a sigmoid-like function with a slower approach to the asymptote (less saturation at large values)
function gompertz(x) #My name's Gomp
    c = -1.0/10000.0
    return ℯ^(-ℯ^(c*x))
end

#-------- Structs and Data Types ------------#
#dictionaries for operations and constants in trees
ϕ = (1.0 + √5.0)/2.0;  #golden ratio, because nature loves the golden ratio
ops_dict = Dict(1 => +, 
                2 => -,
                3 => *,
                4 => safe_divide,
                5 => safe_modulo,
                6 => max,
                7 => min,
                8 => mean
                )
sops_dict = Dict(1 => square, 
                2 => inverse, 
                3 => ln,
                4 => negate,
                5 => floor,
                6 => ceil,
                7 => psqrt,
                8 => abs,
                9 => cube
                ) 

consts_dict = Dict(1 => Float64(π), 2 => 0.0, 3 => 1.0, 4 => 2.0, 5 => 3.0, 6 => ϕ, 7 => Float64(ℯ)) # ℯ is the julia symbol for euler's constant

#pulls an operation on 2 arguments from predefined dictionary
function operation(n, a, b)
    op = ops_dict[n]
    return op(a, b)
end    

#pulls an operation on 1 argument from predefined dictionary
function single_op(n, a)
    sop = sops_dict[n]
    return sop(a)  #It's a mononym - like Cher!
end  

# Leaf Type, one node of tree
mutable struct Leaf
    value
    type
    single_op
    left_child
    right_child
end

# Program Tree Type
struct Program_Tree
    depth::Int64
    root::Leaf
end

# Program_Tree population type - highest level object, to easily facilitate running an evolution experiment
mutable struct Tree_Pop
    pop::Array{Program_Tree}    # population of Program_Trees
    fitnesses::Array{Float64}   # fitnesses associated with each Program_Tree
    elitism::Int64              # number of Program_Trees to preserve as-is to the next generation
    diversity_elitism::Int64    # number of Program_Trees with unique fitnesses to preserve as-is to the next generation
    diversity_generate::Int64   # number of randomly-initialized Program_Trees to add to the population every generation
    fitness_sharing::Bool       # When true, makes Program_Trees with the same fitness split the fitness amongst themselves 
                                    # stops one chromomsome from taking over the population; 
                                    # in evolutionary terms, stops niche-crowding
    selection_algorithm::String # String specifying which selection algorithm (roulette, tournament, ranked, etc) to use in GA
    mutation_rate::Float64      # Float (0 to 1) specifying chance of a child chromosome being mutated
    max_tree_depth::Int64       # max depth of tree
    num_inputs::Int64           # number of possible inputs for trees
    
end

#------------------ Constructors -------------------------- #
#empty constructor for Leaf
function Leaf()
    return Leaf(nothing, nothing, nothing, nothing, nothing)
end

#constructor for Program_Tree type
function Program_Tree(tree_depth::Int64, num_inputs::Int64)
    root = Leaf()
    initialize_tree_topology(tree_depth, 0, root)
    initialize_tree_values(root, num_inputs)

    return Program_Tree(tree_depth, root)
end

#default constructor for Program_Tree population object
function Tree_Pop(pop_size::Int64, elitism::Int64, diversity_elitism::Int64, diversity_generate::Int64, fitness_sharing::Bool, selection_algorithm::String, mutation_rate::Float64, max_tree_depth::Int64, num_inputs::Int64)
    my_pop = [Program_Tree(rand(1:max_tree_depth), num_inputs) for i ∈ 1:pop_size]
    fitnesses = collect(0.0 for i ∈ 1:pop_size)
    return Tree_Pop(my_pop, fitnesses, elitism, diversity_elitism, diversity_generate, fitness_sharing, selection_algorithm, mutation_rate, max_tree_depth, num_inputs)
end

# --------------- Tree Functions -------------------------#
function print_leaf(p::Leaf)
    if p.type == "var"
        if p.single_op !== nothing
            print("[ $(sops_dict[p.single_op])(var$(p.value)) ]\n")
        else
            print("[ var$(p.value) ]\n")
        end
    elseif p.type == "const"
        if p.single_op !== nothing
            print("[ $(sops_dict[p.single_op])($(p.value)) ]\n")
        else
            print("[ $(p.value) ]\n")
        end
    elseif p.type == "op"
        if p.single_op !== nothing
            print("[ $(sops_dict[p.single_op])($(ops_dict[p.value])) ]\n")
        else
            print("[ $(ops_dict[p.value]) ]\n")
        end
    end
end

function print_tree(root::Leaf)
    if root === nothing
        return
    end
    print_leaf(root)
    print_subtree(root, "")
    print("\n")
end

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
function height(leaf)
    if leaf === nothing
        return 0
    end
    left_height = height(leaf.left_child)
    right_height = height(leaf.right_child)
    return max(left_height, right_height) + 1
end

# function to activate tree and compute result
function activate(node::Leaf, inputs::Array)
    if node.type == "const"
        if node.single_op !== nothing
            return single_op(node.single_op, node.value)
        else
            return node.value
        end
    elseif node.type == "var"
        if node.single_op !== nothing
            return single_op(node.single_op, inputs[node.value])
        else
            return inputs[node.value]
        end
    else  # in this case, node is an operation
        #Recursively activate tree
        val = operation(node.value, activate(node.left_child, inputs), activate(node.right_child, inputs))

        if val > 999999999999.999
            return min(val, 999999999999.999)
        else
            return max(val, -999999999999.999)
        end
    end
end

# fill outs tree connections according to depth constraints
function initialize_tree_topology(tree_depth::Int64, depth::Int64, node::Leaf)
    if depth < tree_depth
        if rand() < (1.0 - ((1.0/tree_depth)*depth)/2.0)
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

# intializes a childless leaf
function initialize_childless(node::Leaf, num_inputs::Int64)
    roll = rand()
    if roll < 0.334  #set node to a var
        node.type = "var"
        node.value = rand(1:num_inputs)
    elseif roll < 0.534  #set node to a predefined const
        node.type = "const"
        node.value = consts_dict[rand(1:length(consts_dict))]
    elseif roll < 0.734    #set node to a random int in tight range
        node.type = "const"
        node.value = rand(-12:1:12)
    else  #set node to a random float in wide range
        node.type = "const"
        node.value = rand(-100.0:0.0001:100)
    end

    if rand() < 0.15
        node.single_op = rand(1:length(sops_dict))
    end
end

# initializes a leaf with children (so an operation node)
function initialize_childed(node::Leaf)
    node.value = rand(1:length(ops_dict))
    node.type = "op"
    if rand() < 0.15
        node.single_op = rand(1:length(sops_dict))
    end
end

#copies one tree structure into another
function copy_into(node::Leaf, new_node::Leaf)
    new_node.value = node.value
    new_node.type = node.type
    new_node.single_op = node.single_op
    if node.left_child !== nothing
        new_node.left_child = Leaf()
        copy_into(node.left_child, new_node.left_child)
    end
    if node.right_child !== nothing
        new_node.right_child = Leaf()
        copy_into(node.right_child, new_node.right_child)
    end
end

#----------------- Population Functions ------------------------------#

# function to compute the next population of the genetic algorithm 
#   Using selection, mutation, and crossover operations
function next_generation!(old_pop::Tree_Pop)
    global new_pop = []

    # sort chromosome-fitness pairs in order of fitness
    global sorted_pairs = [[old_pop.pop[i] old_pop.fitnesses[i]] for i ∈ 1:length(old_pop.pop)]
    sort!(sorted_pairs, by=x->x[2], rev=true) #now in descending order
    sorted_fitnesses = [sorted_pairs[i][2] for i ∈ 1:length(old_pop.pop)]

    # preserve chromosomes according to elitism
    for i ∈ 1:old_pop.elitism
        push!(new_pop, sorted_pairs[i][1])
    end

    # preserve chromosomes according to diversity_elitism
    global diversity_count = 0
    global fitness_record = []
    global pop_index = old_pop.elitism + 1
    while diversity_count < old_pop.diversity_elitism && pop_index < length(old_pop.pop)
        global diversity_count
        global fitness_record
        global pop_index

        diversity_fail = false

        for r ∈ fitness_record
            if abs(old_pop.fitnesses[pop_index] - r) < 0.001
                diversity_fail = true
            end
        end

        if diversity_fail == false
            push!(fitness_record, old_pop.fitnesses[pop_index])
            push!(new_pop, old_pop.pop[pop_index])
            diversity_count += 1
        end

        pop_index += 1
    end

    # add new random individuals according to diversity_generate
    for i ∈ 1:old_pop.diversity_generate
        push!(new_pop, Program_Tree(old_pop.max_tree_depth, old_pop.num_inputs))
    end

    normalized_fitnesses = []
    # adjust fitnesses according to fitness_sharing (if enabled)
    if old_pop.fitness_sharing
        for i ∈ 1:length(sorted_pairs)
            push!(normalized_fitnesses, sorted_pairs[i][2]/count(j->(abs(j - sorted_pairs[i][2]) < 0.001), sorted_fitnesses))
        end

        for i ∈ 1:length(sorted_pairs)
            sorted_pairs[i][2] = normalized_fitnesses[i]
        end

        sort!(sorted_pairs, by=x->x[2], rev=true) #re-sorted after normalization
    else
        normalized_fitnesses = sorted_fitnesses
    end

    # perform selection algorithm to select parents for next generation
    if old_pop.selection_algorithm == "roulette"
        global selections = roulette_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop)) 
    elseif old_pop.selection_algorithm == "tournament"
        global selections = tournament_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    elseif old_pop.selection_algorithm == "ranked"
        global selections = ranked_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    elseif old_pop.selection_algorithm == "diversity_search"
        global selections = diversity_search(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    else
        println("GP WARNING | '$(old_pop.selection_algorithm)' not a known selection algorithm. Defaulting to roulette-selection.")
        global selections = roulette_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    end
    
    # perform crossover between parents to create children, mutate with small probability, and populate next generation
    global s_index = 1
    while length(new_pop) < length(old_pop.pop)
        global selections
        global sorted_pairs
        global new_pop
        global s_index

        #crossover to create child
        child = crossover(sorted_pairs[selections[s_index][1]][1], sorted_pairs[selections[s_index][2]][1], old_pop.max_tree_depth)
        
        #mutate child with some probability
        if rand() < old_pop.mutation_rate
            child = mutate(child, old_pop.num_inputs, old_pop.max_tree_depth)
        end
        push!(new_pop, child)    
        s_index += 1
    end

    old_pop.pop = new_pop
    old_pop.fitnesses = [0.0 for i ∈ 1:length(old_pop.fitnesses)]
end

# performs fitness-proportional (aka roulette-wheel) selection and returns indices of selected parents
function roulette_selection(fitnesses, num_selections) 
    selections = []

    #build roulette wheel (from ratios of fitnesses to fitness total)
    roulette_wheel = []
    fitness_proportions = []
    fitness_total = sum(fitnesses)
    for i ∈ 1:length(fitnesses)
        push!(fitness_proportions, fitnesses[i]/fitness_total)
        i == 1 ? push!(roulette_wheel, fitness_proportions[i]) : push!(roulette_wheel, roulette_wheel[i - 1] + fitness_proportions[i])
    end

    for i ∈ 1:num_selections
        spin1 = rand() #random float between 0 and 1
        spin2 = rand() #random float between 0 and 1

        global j = 1
        while roulette_wheel[j] <= spin1
            global j += 1
        end

        global k = 1
        while roulette_wheel[k] <= spin2
            global k += 1
        end

        push!(selections, [j k])
    end

    return selections
end

# performs diversity-search selection and returns indices of selected parents
function diversity_search(fitnesses, num_selections)
    selections = []

    eligible_individuals = []

    #adjust fitnesses (all unique fitnesses equal; all not-unique fitnesses = 0)
    for i ∈ 1:length(fitnesses)
        if count(j->(abs(j - fitnesses[i]) < 0.001), fitnesses) < 2
            push!(eligible_individuals, i)  #congratulations, you're not a basic bitch
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
function tournament_selection(fitnesses, num_selections) 
    selections = []
    k = 2 

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

# performs ranked selection and returns indices of selected parents
function ranked_selection(fitnesses, num_selections) 
    ranked_fitnesses = reverse(collect(1:length(fitnesses)))
    return roulette_selection(ranked_fitnesses, num_selections)
end

#performs crossover operations between trees
function crossover(t1::Program_Tree, t2::Program_Tree, max_tree_depth::Int64)
    
    while true #runs until a child that is not too large is created
        #copy over parents into new objects
        global chrom1 = Program_Tree(t1.depth, Leaf())
        global chrom2 = Program_Tree(t2.depth, Leaf())
        copy_into(t1.root, chrom1.root)
        copy_into(t2.root, chrom2.root)

        #select nodes to cross (segments of trees to transplant)
        global node1 = Leaf()
        global node = chrom1.root
        while node1.value === nothing
            global chrom1
            global node1
            if node === nothing
                node = chrom1.root
            end  
            if rand() < 0.1
                node1 = node
            end
            if rand() < 0.5
                node = node.left_child
            else
                node = node.right_child
            end          
        end

        global node2 = Leaf()
        global node = chrom2.root
        while node2.value === nothing
            global chrom2
            global node2
            if node === nothing
                node = chrom2.root
            end  
            if rand() < 0.1
                node2 = node
            end
            if rand() < 0.5
                node = node.left_child
            else
                node = node.right_child
            end          
        end

        temp = Leaf()
        temp.value = node1.value
        temp.type = node1.type
        temp.single_op = node1.single_op
        temp.left_child = node1.left_child
        temp.right_child = node1.right_child

        node1.value = node2.value
        node1.type = node2.type
        node1.single_op = node2.single_op
        node1.left_child = node2.left_child
        node1.right_child = node2.right_child

        node2.value = temp.value
        node2.type = temp.type
        node2.single_op = temp.single_op
        node2.left_child = temp.left_child
        node2.right_child = temp.right_child

        if node1.type != "op"
            node1.left_child = nothing
            node1.right_child = nothing
        end

        if node2.type != "op"
            node2.left_child = nothing
            node2.right_child = nothing
        end

        if height(chrom1.root) <= max_tree_depth
            return chrom1
        elseif height(chrom2.root) <= max_tree_depth
            return chrom2
        end
    end
end


# mutates a program tree by altering one node at random
function mutate(t::Program_Tree, num_inputs::Int64, max_height::Int64)
    while true
        t2 = Program_Tree(t.depth, Leaf())
        copy_into(t.root, t2.root)
        #select nodes to mutate
        global new_node = Leaf()
        global old_node = t2.root   
        while new_node.value === nothing
            global old_node
            global new_node
            if (old_node === nothing)    
                old_node = t2.root   
            end
            if rand() < 0.1
                new_node = old_node
            end
            if rand() < 0.5
                old_node = old_node.left_child
            else
                old_node = old_node.right_child
            end
        end

        roll = rand()
        if roll < 0.2  #set node to an op
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
            initialize_childless(new_node, num_inputs)
        end

        if height(t2.root) <= max_height
            return t2
        end
    end
end
