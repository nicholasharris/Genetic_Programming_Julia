#test to find expression that correctly computes the area of a circle
include("al_evo.jl")


#=my_AGE = AGE(30, 30, inputs)
println(my_AGE.string_form)

println("result: $(eval(Meta.parse(my_AGE.string_form)))")

A = 3.33333;
B = 3.33333;
C = 3.33333;

println("result2: $(eval(my_AGE.expr_form))")

@time eval(Meta.parse(my_AGE.string_form))
@time eval(my_AGE.expr_form)=#

global stop_cond = false
global areas = [π*(r^2) for r ∈ 1:100]
global R = 0

global index = 1
global lowest_error = 9999999.9
while stop_cond == false
    global index
    global lowest_error
    global areas
    global R
    global stop_cond

    my_AGE = AGE(2,3,['R'])
    error = 0.0
    for i ∈ 1:100
        R = i
        error += abs(areas[i] - eval(my_AGE.expr_form))
    end
    if error < 1.0
        stop_cond = true
        println("AGE found with error $error on attempt $index")
        println("string_form of AGE: " * my_AGE.string_form)
    elseif error < lowest_error
        lowest_error = error
    end
    index % 10 == 0 ? println("index: $index, lowest error thus far: $lowest_error") : "egg"
    index += 1
end
