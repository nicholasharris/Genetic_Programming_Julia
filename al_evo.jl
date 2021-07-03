#--------------------------------------------#
#= Algebraic-Genetic Expressions
    A hair-brained scheme by one Nicholas Harris 
    A.K.A. Genetic Programming but I don't wanna use trees
    Because I'm a little bitch =#

#= Using Julia because I'm a hipster =#
#--------------------------------------------#
using Base: String, Number
using Random

#-------- Structs and Data Types ------------#
#lexicon for valid symbols in chromosomes
ϕ = (1.0 + √5.0)/2.0;  #golden ratio
ops = ["+", "-"];
t_ops = ["*"];
sops = ["square", "inverse", "ln", "negate", "floor", "ceil", "psqrt", "abs"];
consts = ["π", "0.0", "1.0", "2.0", "3.0", "ϕ", "ℯ"];  # ℯ is the julia symbol for euler's constant
inputs = [l for l ∈ 'A':'C']; # num of pre-selected symbols for inputs
A = 6.66666;
B = 6.66666;
C = 6.66666;

# Algebraic-Genetic Expression type
struct AGE
    string_form::String
    expr_form
end

#first contructor for AGE
function AGE(num_terms::Int64, max_term_size::Int64, inputs::Array{Char})
    my_string = "";
    for i ∈ 1:num_terms
        my_string *= Term(max_term_size, inputs).string_form;
        
        i < num_terms ? my_string *= ops[rand(MersenneTwister(), 1:length(ops))] : "egg" #returning the string "egg" does nothing
    end

    return AGE(my_string)
end


#second constructor for AGE
function AGE(s::String)
    return AGE(s, Meta.parse(s))
end

#Term type, mathematical expression that may contain any op except + and -
struct Term
    string_form::String
end
#--------------------------------------------#
#default Term constructor, creates a random term from acceptable consts/inputs/t_ops/sops
function Term(max_term_size::Int64, inputs::Array{Char})
    #20% chance to wrap the whole term in a sop
    my_string = ""
    rand(MersenneTwister(), 1:5) == 5 ? my_string *= sops[rand(MersenneTwister(), 1:length(sops))] : "egg" #returning the string "egg" does nothing

    my_string *= "(";
    term_size = rand(MersenneTwister(), 1:max_term_size);
    for i ∈ 1:term_size 
        if rand(MersenneTwister(), 1:2) == 1  #apply single_op to var/const (50/50 chance)
            #pick sop
            my_string *= sops[rand(MersenneTwister(), 1:length(sops))]
            my_string *= "(" #open sop parenthesis

            #pick input or const to apply sop to (split between input, preset-const, or randomly-generated const)
            roll = rand(MersenneTwister(), 1:3)
            roll == 1 ? my_string *= inputs[rand(MersenneTwister(), 1:length(inputs))] : roll == 2 ? my_string *= consts[rand(MersenneTwister(), 1:length(consts))] : my_string *= string(rand(MersenneTwister(), -100.0:0.001:100))
            
            my_string *= ")" #close sop parenthesis

            i < term_size ? my_string *= t_ops[rand(MersenneTwister(), 1:length(t_ops))] : "egg" #returning the string "egg" does nothing

        else  #case with no sop
            #pick input or const to apply sop to (split between input, preset-const, or randomly-generated const)
            roll = rand(MersenneTwister(), 1:3)
            roll == 1 ? my_string *= inputs[rand(MersenneTwister(), 1:length(inputs))] : roll == 2 ? my_string *= consts[rand(MersenneTwister(), 1:length(consts))] : my_string *= string(rand(MersenneTwister(), -100.0:0.001:100))
            
            i < term_size ? my_string *= t_ops[rand(MersenneTwister(), 1:length(t_ops))] : "egg" #returning the string "egg" does nothing
        end
    end
    my_string *= ")"

    return Term(my_string); #default constructor w/ created term string passed as string_form in the new object
end

#--------- Functions -------------------------#
function square(x)
    return x^2.0;
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

end
#--------------------------------------------#