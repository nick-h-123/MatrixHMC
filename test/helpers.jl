using Revise, Random, Statistics
using DataFrames, GLM
using FiniteDifferences

function fourierPairs(Λ, n, K)
    # Generate all pairs of k-indices which satisfy
    #       k1 + ... + k_n = K
    # for k_i = -Λ,...,Λ.
    # To be used for sums over fourier modes in actions
    # Currently only for n = 2,3

    list = []
    if n == 2
        for i = 1:2*Λ+1
            v_i = ks[i]
            if abs(v_i - K) <= Λ
                append!(list, [[v_i, -v_i + K]])
            end
        end
    end
    if n == 3
        for i = -Λ:Λ
            for j = -Λ:Λ
                if abs(K - i - j) <= Λ
                    newEntry = [i, j, -(i+j) + K]
                    append!(list, [newEntry])
                end
            end
        end
    end
    if n == 4
        for i = -Λ:Λ
            for j = -Λ:Λ
                for k = -Λ:Λ
                    if abs(K - i - j - k) <= Λ
                        newEntry = [i, j, k, -(i + j + k) + K]
                        append!(list, [newEntry])
                    end
                end
            end
        end
    end
    return list
end

"""
include("helpers.jl")
fourPairs = fourierPairs(Λ, 4, 0)
numPairs = length(fourPairs)
fourPairsSort = map(x->map(abs,x), fourPairs)
dict = Dict()
for i = 1:numPairs
    p = fourPairsSort[i]
    if haskey(dict, p)
        dict[p] += 1
    else
        dict[p] = 1
    end
end
fourPairsRed = []
for k in keys(dict)
    append!(fourPairsRed, [(dict[k], k)])
end
(fourPairsRed)

z = [1,2,3,2]
zp = [circshift(z, i) for i=1:length(z)]
fourPairsCPs = copy(fourPairs)
for k = 1:length(fourPairs)
    p = fourPairs[k]
    circPerms = [circshift(p, i) for i=1:length(p)]
    for cp in circPerms
        if cp in fourPairs
            cp_ks = findall(cp_k -> cp_k == cp, fourPairsCPs)
            for cp_k in cp_ks
                if cp_k != k
                    deleteat!(fourPairsCPs, cp_k)
                end
            end
        end
    end
end
"""

function φprod(φ, ks)
    prodRes=1
    for k in ks
        if k < 0
            prodRes *= conj(φ[abs(k)+1])
        else
            prodRes *= φ[k+1]
        end
    end
    return prodRes
end

function running_ave(v)
    vL = length(v)
    v_ave = zeros(vL)
    v_ave[1] = v[1]
    for i = 2:vL
        v_ave[i] = (v[i]+v_ave[i-1]*(i-1))/i
    end
    return v_ave
end

function linreg(x,y)
    data = DataFrame(X=x, Y=y)
    ols = lm(@formula(Y ~ X), data)
    slope = -GLM.coef(ols)[2]
    slope_CI = -GLM.confint(ols)[2,:]
    slope_err = 0.5*(slope_CI[1]-slope_CI[2])
    return slope, slope_err
end

function expectation_value(f, qs, thresh = 0.001)
    # returns the expectation_value of a function f over the samples qs 
    # by finding the number of lags k required for the autocorrelation
    # to cross zero or be less than the threshold thresh. Then, returns
    # average of f over qs skipping over every k samples.
    fs = map(f, qs)
    n_samples = length(qs)
    k=1
    autocorrs = [autocor(fs, [1])[1]]
    while (abs(autocorrs[end]) > thresh || autocorrs[end] > 0) && length(autocorrs) < n_samples    
        append!(autocorrs, autocor(fs, [k])[1])
        k += 1
    end
    if abs(autocorrs[end]) > thresh
        println("Failed to converge")
        return
    end
    sample_is = Array(1:k:n_samples) # array with binned sample indices
    binned_samples = map(i -> fs[i], sample_is) # binned samplers
    # Return mean, variance, and lag k
    return mean(binned_samples), var(binned_samples)/sqrt(length(sample_is)), k
end

function comm(X::AbstractArray,Y::AbstractArray)
    return X*Y-Y*X
end

k1 = 3
k2 = 4
k3 = 5
k4 = 6
p = (1,[k1,k2,k3,k4])
function gen_combos(p)
    k1,k2,k3,k4=p[2]
    combos = [p]
    # commutator properties
    append!(combos, [(-1, [k2,k1,k3,k4])])
    append!(combos, [(-1, [k1,k2,k4,k3])])
    append!(combos, [(+1, [k2,k1,k4,k3])])
    # trace properties
    append!(combos, [(+1, [k3,k4,k1,k2])])
    # trace + comm properties
    append!(combos, [(-1, [k4,k3,k1,k2])])
    append!(combos, [(-1, [k3,k4,k2,k1])])
    append!(combos, [(+1, [k4,k3,k2,k1])])
    return combos
end
combos = gen_combos(p)