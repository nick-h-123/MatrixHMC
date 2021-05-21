function action(u, useDict=true)
    return 0.0
end
# test action
N = 2
Λ = 1
K = 1
metric = MatrixMetric(N, Λ, K)
uTest = rand(metric)/(K*kSize)
ℓπ(u) = -actiοn(u) # save action into log density for AHMC
# Define gradient of log density
function ∂ℓπ∂u(u::AbstractVector, useDict=true)
    return action(u), [zeros(Float64, size(u)...)]
end
# test gradient
# gradtest = ∂ℓπ∂u(uTest)[2]
# if typeof(uTest) != typeof(gradtest) println("GRADIENT MISMATCH") end
# setup HMC
# Define a Hamiltonian system
metric = MatrixMetric(N, Λ, K)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂u)
# Define a leapfrog solver, with initial step size chosen heuristically
initial_u = [map(k->zeros(ComplexF64, N,N),1:kSize)]
rng = MersenneTwister(1234)
initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_u)
#initial_ϵ = find_good_vec_stepsize(rng, hamiltonian, initial_u)
# Define integrator
integrator = Leapfrog(initial_ϵ)
# Define an HMC sampler
# Set number of leapfrog steps
n_steps = 2
# Set the number of warmup iterations
proposal = StaticTrajectory(integrator, n_steps)
#adaptor =  StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))
adaptor = StepSizeAdaptor(0.8, integrator)
# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
progmeter=true
verb=true
#MersenneTwister(1234), 
us, stats = sample(hamiltonian, proposal, initial_u, 
                n_samples+throwaway, adaptor, n_adapts,
                progress=progmeter, verbose=verb)