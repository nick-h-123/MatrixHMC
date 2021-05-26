using MatrixHMC
using Revise, BenchmarkTools, Plots, Random, Statistics
using DataFrames, GLM
using FiniteDifferences
using MathLink, Optim
using StatsBase: autocor, tr
using LaTeXStrings

#include("variational_ON.jl")
include("helpers.jl")
m = 1.0 # mass
β = 10.0/m
Λ = 8 # max momentum index
N = 2 # size of each matric
Kii = 2 # number of bosonic matrices
K = Kii+1 # bosonic + gauge
g = 0.1 # YM coupling
function sample_MT(N, g, m, Λ, n_samples, n_adapts, throwaway; Ki=9)
    β = 10.0/m # total time
    ω = 2*pi/β # momentum spacing
    threePairs = fourierPairs(Λ, 3, 0) # k1 + k2 + k3 = 0
    fourPairs = fourierPairs(Λ, 4, 0) # k1 + k2 + k3 + k4 = 0
    threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ) # k1 + k2 + k3 + k4 = K
    twoPairsK = map(k -> fourierPairs(Λ, 2, k), -Λ:Λ) # k1 + k2 = K
    K = Ki+1 # num of bosonic + one gauge
    kSize = 2*Λ+1 # total num of modes
    # helper objects
    freeFact = β*map(k -> ((ω*k)^2 + m^2), -Λ:Λ) # coefficient for free term
    freeFactA = β*map(k -> (ω*k)^2, -Λ:Λ) # coefficient for free term in A gradient
    # Calculate action:
    function actionAndDicts(u, useDict=true)
        #   u = (X^i, A)
        X = u[1:Ki]
        A = u[K]
        #   freeterm = \sum_k (k^2+m^2) X^i_k X^i_{-k}
        #   commXiXi2term = ([X^i, X^i]^2)_0
        #   commAXi2term = ([A,X^i])_0
        #   dXiCommAXiterm = (dX^i[A,X])
        # free term for X
        function free_term_i(Xi) 
            tr_term = map(k->rtr(Xi[k]*Xi[k], true), 1:kSize)
            return sum(freeFact.*tr_term)
        end
        freeterm = 0.5*sum(map(free_term_i, X))
        # include (∂A)^2 term
        freetermA = 0.5*(sum(map(k->rtr(A[k]*A[k], true), 1:kSize)))
        # initialize
        init_arr = zeros(ComplexF64, length(fourPairs))
        commXiXi2term_arr, commAXi2term_arr, dXiCommAXiterm_arr = init_arr, init_arr, init_arr
        kP = 1
        if g != 0.0
            # initialize dictionaries for trace terms
            commXiXj2Dict, commAXi2Dict = Dict(), Dict()
            commAXi2Dict, dXiCommAXiDict = Dict(), Dict()
            # initialize dictionaries for comm terms:
            #   [X^i_k1, X^j_k2], [A_k1, X^i_k2]
            commXiXjDict, commAXiDict = Dict(), Dict()
            # calculate comm([Xi_k1, Xj_k2]) and store in Dict
            function commXiXj(i, j, k1, k2)
                Xi, Xj = X[i], X[j]
                if useDict
                    key = [i,j,k1,k2]
                    if haskey(commXiXjDict, key)
                        return commXiXjDict[key]
                    else
                        commRes = comm(Xi[k1], Xj[k2])
                        commXiXjDict[key] = commRes
                        if i != j commXiXjDict[[j,i,k1,k2]] = -commRes end
                        return commRes
                    end
                else
                    return comm(Xi[k1], Xj[k2])
                end
            end
            # calculate comm([A_k1, Xi_k2]) and store in Dict
            function commAXi(i, k1, k2)
                Xi = X[i]
                if useDict
                    key = [i,k1,k2]
                    if haskey(commAXiDict, key)
                        return commAXiDict[key]
                    else
                        commRes = comm(A[k1], Xi[k2])
                        commAXiDict[key] = commRes
                        return commRes
                    end
                else
                    return comm(A[k1], Xi[k2])
                end
            end
            # define function for g^2 term
            function calc_g2term()
                for kp in fourPairs
                    k1, k2, k3, k4 = map(k->abs(k)+Λ+1, kp)
                    p = (1,[k1,k2,k3,k4])
                    kcombos = gen_combos(p)
                    if haskey(commXiXj2Dict, p)
                        commXiXi2term_arr[kP] = commXiXj2Dict[p]
                    else
                        termij = map(i -> map(j -> rtr(commXiXj(i, j, k1, k2)*commXiXj(i, j, k3, k4)), 1:Ki), 1:Ki)
                        commXiXi2term_arr_kP = sum(map(sum, termij))
                        if useDict
                            for kcombo in kcombos
                                commXiXj2Dict[kcombo] = kcombo[1]*commXiXi2term_arr_kP
                            end
                        end
                        commXiXi2term_arr[kP] = commXiXi2term_arr_kP
                    end
                    # calculate ([A,X^i]^2)_0 term
                    if haskey(commAXi2Dict, p)
                        commAXi2term_arr[kP] = commAXi2Dict[p]
                    else
                        commAXi2term_arr_kP = sum(map(i -> rtr(commAXi(i, k1, k2)*commAXi(i, k3, k4)), 1:Ki))
                        if useDict
                            for kcombo in kcombos
                                commAXi2Dict[kcombo] = kcombo[1]*commAXi2term_arr_kP
                            end
                        end
                        commAXi2term_arr[kP] = commAXi2term_arr_kP
                    end
                    kP+=1
                end
                g2_term = -0.25*sum(commXiXi2term_arr)-0.5*sum(commAXi2term_arr)
                return β*g^2*g2_term
            end
            g2_term = calc_g2term()
            kP = 1 # reset pair counter
            # define function for g^2 term
            function calc_gterm()
                for kp in threePairs
                    k1, k2, k3 = map(k->abs(k)+Λ+1, kp)
                    p = (1,[k1,k2,k3])
                    kcombos = gen_combos(p)
                    if haskey(dXiCommAXiDict, p)
                        dXiCommAXiterm_arr[kP] = dXiCommAXiDict[p]
                    else
                        dXiCommAXiterm_arr_kP = rtr(sum(map(i -> kp[1]*X[i][k1]*commAXi(i, k2, k3), 1:Ki)), true)
                        if useDict
                            for kcombo in kcombos
                                dXiCommAXiDict[kcombo] = kcombo[1]*dXiCommAXiterm_arr_kP
                            end
                        end
                        dXiCommAXiterm_arr[kP] = dXiCommAXiterm_arr_kP
                    end
                    kP+=1
                end
                return β*g*sum(dXiCommAXiterm_arr)
            end
            g_term = calc_gterm()
            return freeterm  + freetermA + g_term + g2_term, commXiXjDict, commAXiDict # if g != 0
        end
        return freeterm, Dict(), Dict() # if g == 0
    end
    function action(u, useDict=false)
        return real(actionAndDicts(u, useDict)[1])
    end
    # test action
    metric = MatrixMetric(N, Λ, K)
    uTest = rand(metric)/(K*kSize)
    # uTest[end] = map(k->zeros(ComplexF64, N,N), 1:kSize)
    s = action(uTest)
    # println("Im(S)/Re(S) = ", imag(s)/real(s))
    function benchMarkAction()
        println("Beginning benchmark using dictionary...")
        bmDataD = @benchmark action(Xtest, true)
        println("End benchmark using dictionary...")
        println("Beginning benchmark not using dictionary...")
        bmDataND = @benchmark action(Xtest, false)
        println("End benchmark using dictionary...")
        timesD, timesND = bmDataD.times, bmDataND.times
        println("Using Dict:")
        println("\tMean time = ", mean(timesD)/1e6, " ms")
        println("\tStandard deviation = ", std(timesD)/1e6, " ms")
        println("-----------------------------------------------------")
        println("Not using Dict:")
        println("\tMean time = ", mean(timesND)/1e6, " ms")
        println("\tStandard deviation = ", std(timesND)/1e6, " ms")
        println("-----------------------------------------------------")
        println("Performance ratio = ", mean(timesND)/mean(timesD))

        return
    end
    # benchMarkAction()
    ℓπ(u) = -actiοn(u) # save action into log density for AHMC
    # Define gradient of log density
    function ∂ℓπ∂u(u::AbstractVector, useDict=true)
        action_u, commXiXjDict, commAXiDict = actionAndDicts(u, useDict)
        # u = (X^i, A)
        # unpack
        X = u[1:Ki]
        A = u[K]
        # calculate free term for each X^i
        freeTerm = map(Xi->freeFact.*Xi, X)
        # calculate free term for A
        freeTermA = freeFactA.*A
        # freeTermA = map(k->zeros(ComplexF64, N,N), 1:kSize)
        if g == 0
            δSδu = vcat(freeTerm, [freeTermA])
            return -real(action_u), -δSδu
        else
            freeTermA = freeFactA.*A
            # calculate comm([Xi_k1, Xj_k2]) and store in Dict
            function commXiXj(i, j, k1, k2)
                Xi, Xj = X[i], X[j]
                if useDict
                    key = [i,j,k1,k2]
                    if haskey(commXiXjDict, key)
                        return commXiXjDict[key]
                    else
                        commRes = comm(Xi[k1], Xj[k2])
                        commXiXjDict[key] = commRes
                        if i != j commXiXjDict[[j,i,k1,k2]] = -commRes end
                        return commRes
                    end
                else
                    return comm(Xi[k1], Xj[k2])
                end
            end
            # calculate comm([A_k1, Xi_k2]) and store in Dict
            function commAXi(i, k1, k2)
                Xi = X[i]
                if useDict
                    key = [i,k1,k2]
                    if haskey(commAXiDict, key)
                        return commAXiDict[key]
                    else
                        commRes = comm(A[k1], Xi[k2])
                        commAXiDict[key] = commRes
                        return commRes
                    end
                else
                    return comm(A[k1], Xi[k2])
                end
            end
            # initialize interaction term up with g^2 coefficent
            g2_term = map(kk->map(k->zeros(ComplexF64, N,N),1:kSize),1:Ki)
            # array of ([X^j,[X^i,X^j]])_K for K = -Λ,...Λ
            init_arr = map(k->map(kk->zeros(ComplexF64, N,N),1:Ki),1:kSize)
            XjcommXiXj_K = copy(init_arr)
            # array of ([A,[X^i,A]])_K:
            AcommXiA_K = copy(init_arr)
            # array of ([X^i, [A, X^i]]))_K:
            XicommAXi = map(k->zeros(ComplexF64, N,N),1:kSize)
            # begin calculation
            for kp in threePairsK # kp = all 3-pairs adding up to K
                K_kp = sum(kp[1])+Λ+1
                for kpK in kp # kpK = one 3-pair adding up to K
                    k1, k2, k3 = map(k->k+Λ+1, kpK) # unpack pair
                    # calculate ([X^j_k1,[X^i_k2,X^j_k3]])_K
                    XjcommXiXj_K[K_kp] += map(i->sum(map(j->comm(X[j][k1], commXiXj(i,j,k2,k3)), 1:Ki)), 1:Ki) # X^i_k1 for each i
                    # calculate ([A_k1,[X^i_k2,A_k3]])_K
                    AcommXiA_K[K_kp] += map(i->comm(A[k1],-commAXi(i, k2, k1)), 1:Ki)
                    # calculate ([X^i_k1,[A_k2,X^i_k3]])_K
                    XicommAXi[K_kp] += sum(map(i->comm(X[i][k1], commAXi(i, k2, k3)), 1:Ki))
                end
            end
            # reshape terms
            XjcommXiXj = map(i->map(term->term[i], XjcommXiXj_K), 1:Ki)
            AcommXiA = map(i->map(term->term[i], AcommXiA_K), 1:Ki)
            # combine both g^2 term for δSδX
            g2_term = -β*g^2*(XjcommXiXj + AcommXiA)
            # calculate the one g^1 term for δSδX:
            #   (k[X^i_k1, A_k2])_K
            commXiA = copy(init_arr)
            commXiXi = map(k->zeros(ComplexF64, N,N), 1:kSize)
            commAA = map(k->zeros(ComplexF64, N,N), 1:kSize)
            for kp in twoPairsK # kp = all 3-pairs adding up to K
                K_kp = sum(kp[1])+Λ+1 # calculate k1+k2=K
                for kpK in kp # kpK = one 3-pair adding up to K
                    k1, k2 = map(k->k+Λ+1, kpK) # unpack pair
                    # calculate ([X^j_k1,A_k2])_K
                    commXiA[K_kp] += map(i->-commAXi(i, k2, k1), 1:Ki)
                    # calculate ([X^i_k1,X^i_k2])_K
                    commXiXi[K_kp] += kpK[1]*sum(map(i -> commXiXj(i,i,k1,k2), 1:Ki))
                    # calculate (k2 [A_k1,A_k2])_K
                    commAA[K_kp] += kpK[2]*comm(A[k1], A[k2])
                end
                # multiply by momentum factor
                commXiA = 2*sum(kp[1])*commXiA 
            end
            # reshape and multiply by factors
            commXiA = map(i->map(term->term[i], commXiA), 1:Ki)
            # calculate variation w.r.t X
            g_term = β*g*commXiA
            δSδX = freeTerm + g_term + g2_term
            # calculate variation w.r.t A
            g_term_A = β*g*(-commXiXi + commAA)
            g2_term_A = -β*g^2*XicommAXi
            δSδA = freeTermA + g_term_A + g2_term_A
            # combine gradients
            δSδu = vcat(δSδX, [δSδA])
            return -real(action_u), -δSδu
        end
    end
    # test gradient
    # gradtest = ∂ℓπ∂u(uTest)[2][end]
    # if typeof(uTest) != typeof(gradtest) println("GRADIENT MISMATCH") end
    # setup HMC
    # Define a Hamiltonian system
    metric = MatrixMetric(N, Λ, K)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂u)
    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_u = vcat(rand(metric)[1:Ki]/kSize, [map(k->zeros(ComplexF64, N,N),1:kSize)])
    rng = MersenneTwister(1234)
    initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_u)
    #initial_ϵ = find_good_vec_stepsize(rng, hamiltonian, initial_u)
    # Define integrator
    integrator = Leapfrog(initial_ϵ)
    # Define an HMC sampler
    # Set number of leapfrog steps
    n_steps = 1
    # Set the number of warmup iterations
    proposal = StaticTrajectory(integrator, n_steps)
    #adaptor =  StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))
    adaptor = StepSizeAdaptor(0.65, integrator)
    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    progmeter=true
    verb=true
    #MersenneTwister(1234), 
    us, stats = sample(hamiltonian, proposal, initial_u, 
                    n_samples+throwaway, adaptor, n_adapts,
                    progress=progmeter, verbose=verb)
    Xs, As = map(ut->ut[1:Ki], us), map(ut->ut[K], us)
    return Xs, As
end
n_samples = 10
n_adapts = 20
throwaway = 10
Xs, As  = sample_MT(N, g, m, Λ, 
                    n_samples, n_adapts, throwaway;
                    Ki=Kii);
# calculate observables
function getData(Xs, throwaway)
    function twopt_0(Xt)
        bs = sum(map(Xtk->map(k -> real(tr(Xtk[Λ+1+k]*Xtk[Λ+1+k])), 0:Λ), Xt))
        twopt0 = bs[1] + 2*sum(bs[2:end])
        # calculate coefficients for extensions
        d1 = bs[Λ+1]*Λ^2
        d2 = 0
        # extend sum
        n_ext = 1000 # how long to continue extension (could be automated in future)
        twopt0_ext = twopt0 + extendsum(0, d1, d2, β, n_ext)
        return m*twopt0_ext/(Kii*N^2)
    end
    Xkeeps = Xs[throwaway+1:end]
    #twopt_data = map(twopt_0, Xkeeps)
    #twopt_0s = map(x->x[1], twopt_data)
    #twopt_0_exts = map(x->x[2], twopt_data)
    twopt_0s = map(twopt_0, Xkeeps)
    twopt_0s_ave = running_ave(twopt_0s)
    twopts_0_ave = expectation_value(twopt_0, Xkeeps, 0.01)
    return twopts_0_ave, twopt_0s, twopt_0s_ave
end
function extendsum(t, d1, d2, β, Λmax=0)
    if Λmax == 0 Λmax = max(Int(Λ*10), 1000) end
    return sum(map(n -> 2*(d1/n^2+d2/n^4)*cos(n*(2*pi/β)*t), Λ+1:Λmax))
end
# generate data
twopt_0, twopt_0s, twopt_0s_ave = getData(Xs, throwaway)
println("mK<X^i(0)X^i(0)>/N^2 = ", round.(twopt_0[1], sigdigits=5))
println("Effective sample size = ", n_samples/twopt_0[3])
plot(twopt_0s, label="value",
     linewidth=1)
plot!(twopt_0s_ave, label="mean",
      title=string("Equal time 2-point: ", "N = ", N, ", g = ", g),
      linewidth=2, legend = false)
xlabel!("Number of samples")
display(ylabel!(L"\frac{K m}{N^2} \langle tr \left[X^i(0) X^i(0)\right] \rangle"))
#savefig("twopt_MT.png")
# plot(map(Asi->sum(map(Ask->rtr(Ask*Ask),Asi)),As), legend=false)
