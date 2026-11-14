# Fit Interpretable AI's Optimal Policy Trees (OptimalTreePolicyMinimizer) on the tree-check inputs written by
# `python -P scripts/06_tree_check.py --export`, then read back by `python -P scripts/06_tree_check.py`.
#
# Usage, on a machine where IAI runs (it is blocked by Smart App Control on the PC this pipeline was built on):
#     julia +1.11.6/IAI/3.2.2 scripts/06_iai_fit.jl outputs/iai
#
# Each case (a rolling fold x {fixed, budget-aware} tree x depth) gets the same features X and Lagrangian action costs C
# as our search, with max_depth = the case's depth, minbucket = 10 rows and no complexity penalty (cp = 0), so that the
# two solvers differ only in their search. Writes per case:
#   <case>_fitrank.csv   IAI's prescribed treatment on each fitting row (used to check the node table on the Python side)
#   <case>_nodes.csv     the tree: node, leaf, feature, threshold, lower/upper child, treatment ranking (best first)
#   <case>.html          IAI's interactive tree visualisation
using IAI, CSV, DataFrames

dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "outputs", "iai")
cases = CSV.read(joinpath(dir, "cases.csv"), DataFrame)

for row in eachrow(cases)
    c = String(row.case)
    X = CSV.read(joinpath(dir, "$(c)_X.csv"), DataFrame)
    C = CSV.read(joinpath(dir, "$(c)_C.csv"), DataFrame)
    lnr = IAI.OptimalTreePolicyMinimizer(max_depth=row.depth, minbucket=row.minbucket, cp=0.0, random_seed=1)
    t0 = time()
    IAI.fit!(lnr, X, C)
    secs = time() - t0
    rank = IAI.predict_treatment_rank(lnr, X)
    CSV.write(joinpath(dir, "$(c)_fitrank.csv"), DataFrame(first=string.(rank[:, 1])))

    nodes = DataFrame(node=Int[], leaf=Bool[], feature=String[], threshold=Float64[], lower=Int[], upper=Int[],
                      ranking=String[])
    for t in 1:IAI.get_num_nodes(lnr)
        if IAI.is_leaf(lnr, t)
            push!(nodes, (t, true, "", NaN, 0, 0, join(string.(IAI.get_policy_treatment_rank(lnr, t)), " ")))
        else
            IAI.is_parallel_split(lnr, t) || error("case $c, node $t: only parallel (one-feature) splits are expected")
            push!(nodes, (t, false, string(IAI.get_split_feature(lnr, t)), Float64(IAI.get_split_threshold(lnr, t)),
                          IAI.get_lower_child(lnr, t), IAI.get_upper_child(lnr, t), ""))
        end
    end
    CSV.write(joinpath(dir, "$(c)_nodes.csv"), nodes)
    CSV.write(joinpath(dir, "$(c)_time.csv"), DataFrame(seconds=[secs]))
    IAI.write_html(joinpath(dir, "$(c).html"), lnr)
    println("$c: max_depth $(row.depth), $(IAI.get_num_nodes(lnr)) nodes, $(round(secs, digits=1)) s")
end
