# MVP Step 4 (IAI): fit Interpretable AI Optimal Policy Trees on the inputs written by scripts/07_iai_opt.py.
#
#     julia +1.12.6/IAI/3.2.2 scripts/07_iai_fit.jl <dir>
#
# <dir>/cases.csv lists one fit per row: case, fold, fitset (tr or tv), depth, lambda, minbucket.
# For fold F the inputs are <dir>/F_X_<set>.csv (features, columns f0..f7) and <dir>/F_E_<set>.csv (realised model-based
# emissions per action, kg CO2/household, columns a0..aA-1), plus <dir>/F_price.csv (time-average price of each action).
# Action costs = E + lambda * price (the same Lagrangian costs the pipeline's exact tree is fitted on).
# Each fit uses OptimalTreePolicyMinimizer(max_depth = depth, minbucket, cp = 0, random_seed = 1): no complexity penalty,
# so depth is the only complexity control (chosen on validation in Python, as for the exact tree).
# Writes per case: <case>_rank_fit.csv (treatment ranking of each fitting row, best first),
# <case>_rank_eval.csv (ranking of each evaluation row: validation rows for fitset tr, test rows for fitset tv),
# <case>_nodes.csv (tree structure) and <case>_time.csv.
# Note: the IAI system image preloads the IAI module, so it is not listed in `using`.
using CSV, DataFrames

dir = ARGS[1]
cases = CSV.read(joinpath(dir, "cases.csv"), DataFrame)
cache = Dict{String,DataFrame}()
load(name) = get!(() -> CSV.read(joinpath(dir, name), DataFrame), cache, name)
ranks_to_df(r) = DataFrame(Matrix{Int}(parse.(Int, chop.(string.(r), head=1, tail=0))), :auto)

for row in eachrow(cases)
    c, f, s = String(row.case), String(row.fold), String(row.fitset)
    X = load("$(f)_X_$(s).csv")
    E = load("$(f)_E_$(s).csv")
    price = Vector{Float64}(load("$(f)_price.csv").price)
    C = copy(E)
    for (j, nm) in enumerate(names(C))
        C[!, nm] = E[!, nm] .+ row.lambda .* price[j]
    end
    Xeval = load("$(f)_X_$(s == "tr" ? "va" : "te").csv")
    lnr = IAI.OptimalTreePolicyMinimizer(max_depth=row.depth, minbucket=row.minbucket, cp=0.0, random_seed=1)
    t0 = time()
    IAI.fit!(lnr, X, C)
    secs = time() - t0
    CSV.write(joinpath(dir, "$(c)_rank_fit.csv"), ranks_to_df(IAI.predict_treatment_rank(lnr, X)))
    CSV.write(joinpath(dir, "$(c)_rank_eval.csv"), ranks_to_df(IAI.predict_treatment_rank(lnr, Xeval)))
    nodes = DataFrame(node=Int[], leaf=Bool[], feature=String[], threshold=Float64[], lower=Int[], upper=Int[],
                      n=Int[], ranking=String[])
    for t in 1:IAI.get_num_nodes(lnr)
        if IAI.is_leaf(lnr, t)
            push!(nodes, (t, true, "", NaN, 0, 0, IAI.get_num_samples(lnr, t),
                          join(string.(IAI.get_policy_treatment_rank(lnr, t)), " ")))
        else
            IAI.is_parallel_split(lnr, t) || error("case $c, node $t: only parallel (one-feature) splits are expected")
            push!(nodes, (t, false, string(IAI.get_split_feature(lnr, t)), Float64(IAI.get_split_threshold(lnr, t)),
                          IAI.get_lower_child(lnr, t), IAI.get_upper_child(lnr, t), IAI.get_num_samples(lnr, t), ""))
        end
    end
    CSV.write(joinpath(dir, "$(c)_nodes.csv"), nodes)
    CSV.write(joinpath(dir, "$(c)_time.csv"), DataFrame(seconds=[secs]))
    println("$c: depth $(row.depth), lambda $(row.lambda), $(IAI.get_num_nodes(lnr)) nodes, $(round(secs, digits=2)) s")
    flush(stdout)
end
println("done")
