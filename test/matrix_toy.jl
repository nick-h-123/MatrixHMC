using MatrixHMC
using Revise, Plots, Random, Statistics
using DataFrames, GLM
using FiniteDifferences
using MathLink, Optim

#include("variational_ON.jl")
include("helpers.jl")

m = 10.0
β = 10.0/m
ω = 2*pi/β
Λ = 8
N = 3
kSize = 2*Λ+1
# helper objects
freeFact = β*map(k -> ((ω*k)^2 + m^2), -Λ:Λ)
# action and gradient
function action(X)
    free_term = 0.5*sum(map(k-> freeFact[k]*real(tr(X[k]*X[k])), 1:kSize))
    return free_term
end
# test action
metric = MatrixMetric(N, Λ)
Χtest = rand(metric)
action(Χtest)
#
ℓπ(X) = -actiοn(X)
# Define gradient
function ∂ℓπ∂φ(X::AbstractVector)
    freeTerm = map(k -> freeFact[k].*X[k], 1:kSize)
    return -action(X), -freeTerm
end
# test gradient
gradtest = ∂ℓπ∂φ(Χtest)
# setup HMC
# Define a Hamiltonian system
metric = MatrixMetric(N, Λ)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂φ)
# Define a leapfrog solver, with initial step size chosen heuristically
initial_X = rand(metric)
rng = MersenneTwister(1234)
initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_X)   
 # Define integrator
integrator = Leapfrog(initial_ϵ)
# Define an HMC sampler
# Set number of leapfrog steps
n_steps = 5
# Set the number of warmup iterations
proposal = StaticTrajectory(integrator, n_steps)
#adaptor =  StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))
adaptor = StepSizeAdaptor(0.7, integrator)
# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
progmeter=true
verb=true
#MersenneTwister(1234), 
n_samples = 10000
n_adapts = 2000
throwaway = 300
Xs, stats = sample(hamiltonian, proposal, initial_X, 
                   n_samples+throwaway, adaptor, n_adapts,
                   progress=progmeter, verbose=verb)
function getData(Xs)
    bs = map(Xt -> map(k -> real(tr(Xt[Λ+1+k]*Xt[Λ+1+k])), 0:Λ), Xs[throwaway+1:end])
    twopt_0s = map(bsi-> bsi[1] + 2*sum(bsi[2:end]), bs)
    twopt_0s = map(twpt -> twpt, twopt_0s)
    twopt_0s_ave = running_ave(twopt_0s)
    bΛ = map(bs_i -> bs_i[end], bs)
    d1 = bΛ*Λ^2
    d2 = bΛ*0
    twopts_0s_extended = twopt_0s + map(i -> extendsum(0, d1[i], d2[i], β, Int(round(m*1000))), 1:n_samples)
    twopts_0s_extended_ave = running_ave(twopts_0s_extended)
    return m*twopt_0s/N^2, m*twopt_0s_ave/N^2, m*twopts_0s_extended/N^2, m*twopts_0s_extended_ave/N^2
end
function extendsum(t, d1, d2, β, Λmax=0)
    if Λmax == 0 Λmax = max(Int(Λ*10), 1000) end
    return sum(map(n -> 2*(d1/n^2+d2/n^4)*cos(n*(2*pi/β)*t), Λ+1:Λmax))
end
# generate data
twopt_0s, twopt_0s_ave, twopts_0s_ext, twopts_0s_ext_ave = getData(Xs)
plot(twopt_0s, legend=false)
plot(twopt_0s_ave, legend=false)
display(plot!(twopts_0s_ext_ave))


NN = 5
Λ = 2
metric = MatrixMetric(NN, Λ)
s = 0
for i = 1:n_samples
    temp = rand(metric)
    s += sum(map(tempi->real(tr(tempi*tempi)), temp))
end
s/(NN^2*(2*Λ+1))/n_samples

fourPairs = fourierPairs(Λ, 4, 0)
numPairs = length(fourPairs)
#fourPairsRed = Array{Tuple, 1}(undef, numPairs)
fourPairsRed = map(p->map(abs, p), fourPairs)
