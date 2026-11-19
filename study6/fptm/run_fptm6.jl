# Study 6 FPTM run with model logging. Reference FPTM (FPTMVariants.jl with MODE = 0, bit-exact to upstream).
# julia -t 1 run_fptm6.jl <datadir> <ds> <clauses_num> <T> <S> <L> <LF> <seed> <epochs> <every>
# Prints:  RESULT <val_curve> <test> <train_s> <s_int>
#          STATS  <single-line JSON with model structure and quantized-vote test accuracies>
include(joinpath(@__DIR__, "FPTMVariants.jl"))
using .FuzzyPatternTM: TMInput, TMClassifier, train!, predict, accuracy, check_clause_graded
using Random
DD = ARGS[1]; ds = ARGS[2]; CL, T, S, L, LF, seed, EP, EV = parse.(Int, ARGS[3:10])
FuzzyPatternTM.MODE[] = 0
D = joinpath(DD, ds) * "/"
nf, ntr, nva, nte, K = parse.(Int, split(strip(read(D * "meta.txt", String))))
nb = cld(nf, 8)
function loadX(f, n)
    raw = read(D * f); @assert length(raw) == n * nb
    X = Vector{TMInput}(undef, n)
    for r in 1:n
        b = BitVector(undef, nf); o = (r - 1) * nb
        @inbounds for i in 1:nf
            b[i] = (raw[o + ((i - 1) >> 3) + 1] >> ((i - 1) & 7)) & 0x01 == 0x01
        end
        X[r] = TMInput(b)
    end
    X
end
loady(f) = Int16.(reinterpret(Int16, read(D * f)))
Xtr, Xva, Xte = loadX("Xtr.bits", ntr), loadX("Xva.bits", nva), loadX("Xte.bits", nte)
ytr, yva, yte = loady("ytr.i16"), loady("yva.i16"), loady("yte.i16")
Random.seed!(seed)
tm = TMClassifier{Int16}(CL, T, S, L=L, LF=LF, states_num=256, include_limit=128)
curve = Float64[]
t0 = time()
for e in 1:EP
    train!(tm, Xtr, ytr, shuffle=true)
    if e % EV == 0 || e == EP
        push!(curve, round(accuracy(predict(tm, Xva), yva), digits=4))
    end
end
tt = time() - t0
pred = predict(tm, Xte)
te = accuracy(pred, yte)
println("RESULT ", join(string.(curve), ","), " ", te, " ", round(tt, digits=1), " ", tm.s)

# ---- model structure ----
used_feat = falses(nf); used_pos = falses(nf); used_neg = falses(nf)
lens = Int[]
for (cls, ta) in tm.clauses
    for (A, B) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                   (ta.negative_included_literals, ta.negative_included_literals_inverted))
        for i in eachindex(A)
            n = length(A[i]) + length(B[i]); push!(lens, n)
            for k in A[i]; used_feat[k] = true; used_pos[k] = true; end
            for k in B[i]; used_feat[k] = true; used_neg[k] = true; end
        end
    end
end
# ---- quantized vote: clause score clipped at Q = Q threshold searches per clause ----
function qpredict(Q)
    c = 0
    for r in eachindex(Xte)
        best = typemin(Int); bc = nothing
        for (cls, ta) in tm.clauses
            v = 0
            for i in eachindex(ta.positive_included_literals)
                v += min(check_clause_graded(Xte[r], ta.positive_included_literals[i], ta.positive_included_literals_inverted[i], LF), Q)
            end
            for i in eachindex(ta.negative_included_literals)
                v -= min(check_clause_graded(Xte[r], ta.negative_included_literals[i], ta.negative_included_literals_inverted[i], LF), Q)
            end
            if v > best; best = v; bc = cls; end
        end
        c += (bc == yte[r])
    end
    c / length(yte)
end
qs = sort(unique(min.([1, 2, 3, 4, 8, 16, 32, 64], LF)))
qacc = [round(qpredict(q), digits=6) for q in qs]
# score histogram on the test set (fraction of clause evaluations at each score 0..LF)
hist = zeros(Int, LF + 1)
for r in eachindex(Xte), (cls, ta) in tm.clauses
    for i in eachindex(ta.positive_included_literals)
        hist[check_clause_graded(Xte[r], ta.positive_included_literals[i], ta.positive_included_literals_inverted[i], LF) + 1] += 1
    end
    for i in eachindex(ta.negative_included_literals)
        hist[check_clause_graded(Xte[r], ta.negative_included_literals[i], ta.negative_included_literals_inverted[i], LF) + 1] += 1
    end
end
js(v) = "[" * join(string.(v), ",") * "]"
println("STATS {\"nf\":", nf, ",\"K\":", K, ",\"clauses_total\":", length(lens),
        ",\"used_features\":", count(used_feat), ",\"used_pos_literals\":", count(used_pos), ",\"used_neg_literals\":", count(used_neg),
        ",\"clause_len_mean\":", round(sum(lens) / max(1, length(lens)), digits=3), ",\"clause_len_max\":", maximum(lens; init=0),
        ",\"clause_len\":", js(lens), ",\"empty_clauses\":", count(==(0), lens),
        ",\"q\":", js(qs), ",\"q_test\":", js(qacc), ",\"score_hist\":", js(hist), "}")
