module MatrixHMC

const DEBUG = convert(Bool, parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using Statistics: mean, var, middle
using LinearAlgebra: tr, Hermitian, Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
using StatsFuns: logaddexp, logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter: ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck
using StatsBase: autocor

using DocStringExtensions

import StatsBase: sample
import Parameters: reconstruct

include("utilities.jl")

# Notations
# ℓπ: log density of the target distribution
# θ: position variables / model parameters
# ∂ℓπ∂θ: gradient of the log density of the target distribution w.r.t θ
# r: momentum variables
# z: phase point / a pair of θ and r

include("metric.jl")
export UnitEuclideanMetric, EuclideanMetric, HermitianMetric, MatrixMetric

include("hamiltonian.jl")
export Hamiltonian

include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog

include("trajectory.jl")
export Trajectory, HMCKernel,
       FixedNSteps, FixedIntegrationTime,
       ClassicNoUTurn, GeneralisedNoUTurn, StrictGeneralisedNoUTurn,
       EndPointTS, SliceTS, MultinomialTS, 
       find_good_stepsize, find_good_vec_stepsize

abstract type AbstractTrajectory end

struct StaticTrajectory{TS} end
@deprecate StaticTrajectory{TS}(int::AbstractIntegrator, L) where {TS} HMCKernel(Trajectory{TS}(int, FixedNSteps(L)))
@deprecate StaticTrajectory(int::AbstractIntegrator, L) HMCKernel(Trajectory{EndPointTS}(int, FixedNSteps(L)))
@deprecate StaticTrajectory(ϵ::AbstractScalarOrVec{<:Real}, L) HMCKernel(Trajectory{EndPointTS}(Leapfrog(ϵ), FixedNSteps(L)))

struct HMCDA{TS} end
@deprecate HMCDA{TS}(int::AbstractIntegrator, λ) where {TS} HMCKernel(Trajectory{TS}(int, FixedIntegrationTime(λ)))
@deprecate HMCDA(int::AbstractIntegrator, λ) HMCKernel(Trajectory{MetropolisTS}(int, FixedIntegrationTime(λ)))
@deprecate HMCDA(ϵ::AbstractScalarOrVec{<:Real}, λ) HMCKernel(Trajectory{MetropolisTS}(Leapfrog(ϵ), FixedIntegrationTime(λ)))

@deprecate find_good_eps find_good_stepsize

export StaticTrajectory, HMCDA, find_good_eps

include("adaptation/Adaptation.jl")
using .Adaptation
import .Adaptation: StepSizeAdaptor, MassMatrixAdaptor, StanHMCAdaptor, NesterovDualAveraging

# Helpers for initializing adaptors via AHMC structs

StepSizeAdaptor(δ::AbstractFloat, stepsize::AbstractScalarOrVec{<:AbstractFloat}) = 
    NesterovDualAveraging(δ, stepsize)
StepSizeAdaptor(δ::AbstractFloat, i::AbstractIntegrator) = StepSizeAdaptor(δ, nom_step_size(i))

MassMatrixAdaptor(m::UnitEuclideanMetric{T}) where {T} =
    UnitMassMatrix{T}()
MassMatrixAdaptor(m::EuclideanMetric{T}) where {T} =
    WelfordVar{T}(size(m); var=copy(m.M⁻¹))
MassMatrixAdaptor(m::HermitianMetric{T}) where {T} =
    WelfordCov{T}(size(m); cov=copy(m.M⁻¹))

MassMatrixAdaptor(
    m::Type{TM},
    sz::Tuple{Vararg{Int}}=(2,)
) where {TM<:AbstractMetric} = MassMatrixAdaptor(Float64, m, sz)

MassMatrixAdaptor(
    ::Type{T},
    ::Type{TM},
    sz::Tuple{Vararg{Int}}=(2,)
) where {T, TM<:AbstractMetric} = MassMatrixAdaptor(TM(T, sz))

# Deprecations

@deprecate StanHMCAdaptor(n_adapts, pc, ssa) initialize!(StanHMCAdaptor(pc, ssa), n_adapts)
@deprecate NesterovDualAveraging(δ::AbstractFloat, i::AbstractIntegrator) StepSizeAdaptor(δ, i)
@deprecate Preconditioner(args...) MassMatrixAdaptor(args...)

export StepSizeAdaptor, NesterovDualAveraging, 
       MassMatrixAdaptor, UnitMassMatrix, WelfordVar, WelfordCov, 
       NaiveHMCAdaptor, StanHMCAdaptor

include("diagnosis.jl")

include("sampler.jl")
export sample

include("contrib/ad.jl")
end # module
