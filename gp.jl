#--------------------------------------------#
#=  Genetic Programming with Julia 
        by Nick Harris              =#
#--------------------------------------------#
using Base: String, Number, Int64, Bool, Symbol
using Random
using StatsBase: countmap

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
    return √(abs(x)) #keep tricksy false imaginaries away!
end

function sigmoid(x)
    return 1.0/(1 + ℯ^(-x))
end

#gompertz is a sigmoid-like function with a slower approach to the asymptote (less saturation at large values)
function gompertz(x) #My name's Gomp
    c = -1.0/10000.0
    return ℯ^(-ℯ^(c*x))
end

#--------------- Structs, Data Types, and Predefined Objects -------------------#
# dictionaries for operations and constants in trees
ϕ = (1.0 + √5.0)/2.0;  #golden ratio, because nature loves the golden ratio
ops_dict = Dict(1 => +, 
                2 => -,
                3 => *,
                4 => safe_divide,
                5 => safe_modulo,
                6 => max,
                7 => min,
                8 => mean)
sops_dict = Dict(1 => square, 
                2 => inverse, 
                3 => ln,
                4 => negate,
                5 => floor,
                6 => ceil,
                7 => psqrt,
                8 => abs,
                9 => cube) 
consts_dict = Dict(1 => Float64(π), 
                   2 => 0.0, 
                   3 => 1.0, 
                   4 => 2.0, 
                   5 => 3.0, 
                   6 => ϕ, 
                   7 => Float64(ℯ)) # ℯ is the julia symbol for euler's constant

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
    num_children::Int64
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
    k_value::Int64              # optional keyword argument, k-value for tournament selection
end

#--------------------------- Constructors ---------------------------------- #
#empty constructor for Leaf
function Leaf()
    return Leaf(nothing, nothing, nothing, nothing, nothing, 0)
end

#constructor for Program_Tree type
function Program_Tree(tree_depth::Int64, num_inputs::Int64)
    root = Leaf()
    initialize_tree_topology(tree_depth, 1, root)
    insert_children_count(root)
    initialize_tree_values(root, num_inputs)

    return Program_Tree(tree_depth, root)
end

#default constructor for Program_Tree population object
function Tree_Pop(pop_size::Int64, elitism::Int64, diversity_elitism::Int64, diversity_generate::Int64, fitness_sharing::Bool, selection_algorithm::String, mutation_rate::Float64, max_tree_depth::Int64, num_inputs::Int64; k = 2)
    my_pop = [Program_Tree(rand(1:max_tree_depth), num_inputs) for i ∈ 1:pop_size]
    fitnesses = collect(0.0 for i ∈ 1:pop_size)
    return Tree_Pop(my_pop, fitnesses, elitism, diversity_elitism, diversity_generate, fitness_sharing, selection_algorithm, mutation_rate, max_tree_depth, num_inputs, k)
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
function get_elements(root)
    if root === nothing
        return 0
    end
    return (get_elements(root.left_child) + get_elements(root.right_child) + 1)
end

# inserts children count for each node
function insert_children_count(root)
    if root === nothing
        return
    end
    root.num_children = get_elements(root) - 1
    insert_children_count(root.left_child)
    insert_children_count(root.right_child)
end

# returns number of children for root
function num_children(root)
    if root === nothing
        return 0
    end
    return root.num_children + 1   #Why does num_children() return a different value than someLeaf.num_children ?
                                        # I dunno, got this shitty code off the internet to make the random_node function work
end

# helper function to return a random node
function random_node_util(root, count)
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
function random_node(root)
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

        if node.single_op !== nothing
            val = single_op(node.single_op, val) 
        end

        if val > 999999999999.999
            return min(val, 999999999999.999)
        else
            return max(val, -999999999999.999)
        end
    end
end

# intializes a childless leaf
function initialize_childless(node::Leaf, num_inputs::Int64)
    roll = rand()
    if roll < 0.25  #set node to a var
        node.type = "var"
        node.value = rand(1:num_inputs)
    elseif roll < 0.5  #set node to a predefined const
        node.type = "const"
        node.value = consts_dict[rand(1:length(consts_dict))]
    elseif roll < 0.75    #set node to a random int in tight range
        node.type = "const"
        node.value = rand(-12:1:12)
    else  #set node to a random float in wide range
        node.type = "const"
        node.value = rand(-100.0:0.0001:100)
    end

    if rand() < 0.25
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

#copies one tree structure into another
function copy_into(node::Leaf, new_node::Leaf)
    new_node.value = node.value
    new_node.type = node.type
    new_node.single_op = node.single_op
    new_node.num_children = node.num_children
    if node.left_child !== nothing
        new_node.left_child = Leaf()
        copy_into(node.left_child, new_node.left_child)
    end
    if node.right_child !== nothing
        new_node.right_child = Leaf()
        copy_into(node.right_child, new_node.right_child)
    end
end

#prints program tree to a file so it can be saved and recovered later
function save_tree(node, outfile::IOStream)
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
function load_tree!(node, infile::IOStream)
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
            node.single_op = parse(Int64, element_list[1])
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
    global new_pop = []

    # sort chromosome-fitness pairs in order of fitness
    global sorted_pairs = [[old_pop.pop[i] old_pop.fitnesses[i]] for i ∈ 1:length(old_pop.pop)]
    sort!(sorted_pairs, by=x->x[2], rev=true) #now in descending order
    sorted_fitnesses = [round(sorted_pairs[i][2], digits = 6) for i ∈ 1:length(old_pop.pop)]
    global counter = countmap(sorted_fitnesses, alg= :dict)

    # preserve chromosomes according to elitism
    for i ∈ 1:old_pop.elitism
        push!(new_pop, sorted_pairs[i][1])
    end

    # preserve chromosomes according to diversity_elitism
    global diversity_count = 0
    global pop_index = 1
    while diversity_count < old_pop.diversity_elitism && pop_index < length(old_pop.pop)
        global diversity_count
        global pop_index
        global counter

        push!(new_pop, sorted_pairs[pop_index][1])
        diversity_count += 1
        pop_index += counter[sorted_fitnesses[pop_index]]
    end

    # add new random individuals according to diversity_generate
    for i ∈ 1:old_pop.diversity_generate
        if length(new_pop) < length(old_pop.pop)
            push!(new_pop, Program_Tree(old_pop.max_tree_depth, old_pop.num_inputs))
        end
    end

    normalized_fitnesses = []
    # adjust fitnesses according to fitness_sharing (if enabled)
    if old_pop.fitness_sharing
        for i ∈ 1:length(sorted_fitnesses)
            if sorted_fitnesses[i] >= 0
                push!(normalized_fitnesses, sorted_fitnesses[i]/counter[sorted_fitnesses[i]])
            else
                push!(normalized_fitnesses, sorted_fitnesses[i]*counter[sorted_fitnesses[i]])
            end
        end
        sort!(normalized_fitnesses, rev=true) #re-sorted after normalization
    else
        normalized_fitnesses = sorted_fitnesses
    end

    # perform selection algorithm to select parents for next generation
    if old_pop.selection_algorithm == "roulette"
        global selections = roulette_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop)) 
    elseif old_pop.selection_algorithm == "tournament"
        global selections = tournament_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop),  old_pop.k_value)
    elseif old_pop.selection_algorithm == "ranked"
        global selections = ranked_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    elseif old_pop.selection_algorithm == "diversity_search"
        global selections = diversity_search(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    elseif old_pop.selection_algorithm == "SUS"
        global selections = stochastic_universal_sampling(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop))
    else
        println("GP WARNING | '$(old_pop.selection_algorithm)' not a known selection algorithm. Defaulting to tournament-selection.")
        global selections = tournament_selection(normalized_fitnesses, length(normalized_fitnesses) - length(new_pop),  old_pop.k_value)
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

    points = sort([rand() for i ∈ 1:(2*num_selections)])
    flat_selections = []

    global j = 1 
    for i ∈ 1:(2*num_selections)       
        while roulette_wheel[j] < points[i]
            global j
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
    for i ∈ 1:length(fitnesses)
        push!(fitness_proportions, fitnesses[i]/fitness_total)
        i == 1 ? push!(roulette_wheel, fitness_proportions[i]) : push!(roulette_wheel, roulette_wheel[i - 1] + fitness_proportions[i])
    end

    N = 2*num_selections
    p_distance = 1.0/N
    start = rand()*p_distance
    points = [start + i*p_distance for i ∈ 0:(N-1)]

    flat_selections = []

    global j = 1 
    for i ∈ 1:N       
        while roulette_wheel[j] < points[i]
            global j
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
function tournament_selection(fitnesses, num_selections, k_value) 
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

# performs ranked selection and returns indices of selected parents
function ranked_selection(fitnesses, num_selections) 
    ranked_fitnesses = reverse(collect(1:length(fitnesses)))
    return roulette_selection(ranked_fitnesses, num_selections)
end

# performs crossover operations between trees
function crossover(t1::Program_Tree, t2::Program_Tree, max_tree_depth::Int64)
    while true #runs until a child that is not too large is created
        #copy over parents into new objects
        global chrom1 = Program_Tree(t1.depth, Leaf())
        global chrom2 = Program_Tree(t2.depth, Leaf())
        copy_into(t1.root, chrom1.root)
        copy_into(t2.root, chrom2.root)

        #select nodes to cross (segments of trees to transplant)
        node1 = random_node(chrom1.root)
        node2 = random_node(chrom2.root)

        #cross nodes
        temp = Leaf()
        temp.type = node1.type
        temp.value = node1.value
        temp.single_op = node1.single_op
        temp.left_child = node1.left_child
        temp.right_child = node1.right_child

        node1.type = node2.type
        node1.value = node2.value
        node1.single_op = node2.single_op
        node1.left_child = node2.left_child
        node1.right_child = node2.right_child

        node2.type = temp.type
        node2.value = temp.value
        node2.single_op = temp.single_op
        node2.left_child = temp.left_child
        node2.right_child = temp.right_child

        #alter nodes after crossing to keep the program tree valid
        if node1.type != "op"
            node1.left_child = nothing
            node1.right_child = nothing
        end

        if node2.type != "op"
            node2.left_child = nothing
            node2.right_child = nothing
        end

        #return if resulting tree does not exceed height limit
        if height(chrom1.root) <= max_tree_depth
            insert_children_count(chrom1.root)
            return chrom1
        elseif height(chrom2.root) <= max_tree_depth
            insert_children_count(chrom2.root)
            return chrom2
        end
    end
end

# mutates a program tree by altering one node at random
function mutate(t::Program_Tree, num_inputs::Int64, max_height::Int64)
    if rand() < 0.5 # strong mutate: completely replace node with random one  
        while true
            t2 = Program_Tree(t.depth, Leaf())
            copy_into(t.root, t2.root)
            
            #select node to mutate
            new_node = random_node(t2.root)

            roll = rand()
            if roll < 0.5  #set node to an op
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
                insert_children_count(t2.root)
                return t2
            end
        end
    else  #weak mutate, just modify one component of node
        t2 = Program_Tree(t.depth, Leaf())
        copy_into(t.root, t2.root)
        new_node = random_node(t2.root)

        roll = rand()
        if roll < 0.25 #modify sop
            if rand() < 0.5
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
                roll2 = rand()
                if roll2 < 0.3333333
                    new_node.value = consts_dict[rand(1:length(consts_dict))]
                elseif roll2 < 0.6666666
                    new_node.value = rand(-12:1:12)
                else
                    new_node.value = rand(-100.0:0.0001:100)
                end
                       
            end
        end

        return t2
    end
end