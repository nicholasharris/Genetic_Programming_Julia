# ------------- Elementary functions for use in program trees -------------- #
function avg(a, b)
    return (a + b)/2
end

function psqrt(x)
    return √(abs(x)) # keep tricksy false imaginaries away!
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

function lc90(a, b)
    return a*0.9 + b*0.1
end

function lc80(a, b)
    return a*0.8 + b*0.2
end

function lc70(a, b)
    return a*0.7 + b*0.3
end

function lc60(a, b)
    return a*0.6 + b*0.4
end

function lc50(a, b)
    return a*0.5 + b*0.5
end

function lc55(a, b)
    return a*0.55 + b*0.45
end

function lc65(a, b)
    return a*0.65 + b*0.35
end

function lc75(a, b)
    return a*0.75 + b*0.25
end

function lc85(a, b)
    return a*0.85 + b*0.15
end

function lc95(a, b)
    return a*0.95 + b*0.05
end

function left(a, b)
    return a
end

function right(a, b)
    return b
end

function square(x)
    return x^2
end

function cube(x)
    return x^3
end

function cube_sa(l)
    return 6*(l^2)
end

function cube_sa_rt(a)
    return psqrt(a/6)
end

function circle(r)
    return 3.1415926535897*(r^2)
end

function circle_rt(a)
    return psqrt(a/3.1415926535897)
end

function hexagon(r)
    return ((3*psqrt(3))/2)*r^2
end

function hexagon_rt(a)
    return psqrt((2*a)/(3*psqrt(3)))
end

function sphere(r)
    return (4.0/3.0)*3.1415926535897*(r^3)
end

function sphere_rt(v)
    return cbrt(v/((4/3)*3.1415926535897))
end

function pyramid(l)
    return (l^3)/3.0
end

function pyramid_rt(v)
    return cbrt(3.0*v)
end

function tetrahedron(l)
    return (l^3)/(6*psqrt(2))
end

function tetrahedron_rt(v)
    return cbrt(6*psqrt(2)*v)
end

function octahedron(l)
    return (psqrt(2)/3)*(l^3)
end

function octahedron_rt(v)
    return cbrt((3*v)/psqrt(2))
end

function dodecahedron(l)
    return ((15 + 7*psqrt(5))/4)*(l^3)
end

function dodecahedron_rt(v)
    return cbrt((4*v)/(15 + 7*psqrt(5)))
end

function icosahedron(l)
    return ((5*(3 + psqrt(5)))/12)*(l^3)
end

function icosahedron_rt(v)
    return cbrt((12*v)/(5*(3 + psqrt(5))))
end

function negate(x)
    return -1.0 * x
end

function inverse(x)
    return abs(x) > 0.001 ? 1.0/x : 1.0
end

function ln(x)
    return x > 0.001 ? log(x) :  1.0
end

function log_10(x)
    return x > 0.001 ? log(10, x) : 1.0
end

function log_2(x)
    return x > 0.001 ? log(2, x) : 1.0
end

function sigmoid(x)
    return 1.0/(1 + 2.7182818284590^(-x))
end

function sawtooth(x)
    return x % 1
end

function max_magnitude(x, y)
    return max(abs(x), abs(y))
end

function min_magnitude(x, y)
    return min(abs(x), abs(y))
end

function avg_magnitude(x, y)
    return avg(abs(x), abs(y))
end

# gompertz is a sigmoid-like function with a slower approach to the asymptote (less saturation at large values)
function gompertz(x)
    c = -1.0/10000.0
    return 2.7182818284590^(-2.7182818284590^(c*x))
end

# dictionaries for operations and constants in trees
ϕ = (1.0 + √5.0)/2.0;  #golden ratio, because nature loves the golden ratio
const ops_dict = Dict(1 => +, 
                2 => -,
                3 => *,
                4 => safe_divide,
                5 => safe_modulo,
                6 => max,
                7 => min,
                8 => avg,
                9 => hypot,
                10 => lc90,
                11 => lc80,
                12 => lc70,
                13 => lc60, 
                14 => lc55,
                15 => lc65,
                16 => lc75,
                17 => lc85,
                18 => lc95,
                19 => left,
                20 => right,
                21 => max_magnitude,
                22 => min_magnitude,
                23 => avg_magnitude,
                )
const sops_dict = Dict( 1 => square, 
                2 => inverse, 
                3 => ln,
                4 => negate,
                5 => floor,
                6 => ceil,
                7 => psqrt,
                8 => abs,
                9 => cube,
                10 => mod2pi,
                11 => cbrt,
                12 => sin,
                13 => cos,
                14 => circle,
                15 => sphere,
                16 => circle_rt,
                17 => sphere_rt,
                18 => sigmoid,
                19 => gompertz,
                20 => log_10,
                21 => log_2,
                22 => tanh,
                23 => sech,
                24 => pyramid,
                25 => pyramid_rt,
                26 => tetrahedron,
                27 => tetrahedron_rt,
                28 => octahedron,
                29 => octahedron_rt,
                30 => dodecahedron,
                31 => dodecahedron_rt,
                32 => icosahedron,
                33 => icosahedron_rt,
                34 => hexagon,
                35 => hexagon_rt,
                36 => cube_sa,
                37 => cube_sa_rt,
                38 => sawtooth,
                39 => round  
                )

const consts_dict = Dict(1 => Float64(π), 
                   2 => 0.0, 
                   3 => 1.0, 
                   4 => 2.0, 
                   5 => 3.0, 
                   6 => ϕ, 
                   7 => Float64(ℯ)) # ℯ is the julia symbol for euler's constant

#pulls an operation on 2 arguments from predefined dictionary
function operation(n, a, b)
   
    @match n begin
        1 => return a + b
        2 => return a - b
        3 => return a * b
        4 => return safe_divide(a, b)
        5 => return safe_modulo(a, b)
        6 => return max(a, b)
        7 => return min(a, b)
        8 => return avg(a, b)
        9 => return hypot(a, b)
        10 => return lc90(a, b)
        11 => return lc80(a, b)
        12 => return lc70(a, b)
        13 => return lc60(a, b)
        14 => return lc55(a, b)
        15 => return lc65(a, b)
        16 => return lc75(a, b)
        17 => return lc85(a, b)
        18 => return lc95(a, b)
        19 => return left(a, b)
        20 => return right(a, b)
        21 => return max_magnitude(a, b)
        22 => return min_magnitude(a, b)
        23 => return avg_magnitude(a, b)
        _ => return "-"
    end
end    


#pulls an operation on 1 argument from predefined dictionary
function single_op(n, a)  
    
    @match n begin
        1 => return square(a) 
        2 => return inverse(a)
        3 => return ln(a)
        4 => return negate(a)
        5 => return floor(a)
        6 => return ceil(a)
        7 => return psqrt(a)
        8 => return abs(a)
        9 => return cube(a)
        10 => return mod2pi(a)
        11 => return cbrt(a)
        12 => return sin(a)
        13 => return cos(a)
        14 => return circle(a)
        15 => return sphere(a)
        16 => return circle_rt(a)
        17 => return sphere_rt(a)
        18 => return sigmoid(a)
        19 => return gompertz(a)
        20 => return log_10(a)
        21 => return log_2(a)
        22 => return tanh(a)
        23 => return sech(a)
        24 => return pyramid(a)
        25 => return pyramid_rt(a)
        26 => return tetrahedron(a)
        27 => return tetrahedron_rt(a)
        28 => return octahedron(a)
        29 => return octahedron_rt(a)
        30 => return dodecahedron(a)
        31 => return dodecahedron_rt(a)
        32 => return icosahedron(a)
        33 => return icosahedron_rt(a)
        34 => return hexagon(a)
        35 => return hexagon_rt(a)
        36 => return cube_sa(a)
        37 => return cube_sa_rt(a)
        38 => return sawtooth(a)
        39 => return round(a)
        _ => return "wrong type :: $n, $(typeof(n))"
       
    end
end  
