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