using ITensors, ITensorMPS
using OptimKit
using Random
using Zygote

nsites = 4 # Number of sites
nlayers = 2 # Layers of gates in the ansatz
gradtol = 1e-4 # Tolerance for stopping gradient descent

# The Hamiltonian we are minimizing
function ising_hamiltonian(nsites; h)
  ℋ = OpSum()
  for j in 1:(nsites-1)
    ℋ += -1, "Z", j, "Z", j + 1
  end
  for j in 1:nsites
    ℋ += h, "X", j
  end
  return ℋ
end

# A layer of the circuit we want to optimize
function layer(nsites, θ⃗)
  RY_layer = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:nsites]
  CX_layer = [("CX", (n, n + 1)) for n in 1:2:(nsites-1)]
  return [RY_layer; CX_layer]
end

# The variational circuit we want to optimize
function variational_circuit(nsites, nlayers, θ⃗)
  range = 1:nsites
  circuit = layer(nsites, θ⃗[range])
  for n in 1:(nlayers-1)
    circuit = [circuit; layer(nsites, θ⃗[range .+ n * nsites])]
  end
  return circuit
end

s = siteinds("Qubit", nsites)

h = 1.3
ℋ = ising_hamiltonian(nsites; h=h)
H = MPO(ℋ, s)
ψ0 = MPS(s, "0")

#
# The loss function, a function of the gate parameters
# and implicitly depending on the Hamiltonian and state:
#
# loss(θ⃗) = ⟨0|U(θ⃗)† H U(θ⃗)|0⟩ = ⟨θ⃗|H|θ⃗⟩
#
function loss(θ⃗)
  nsites = length(ψ0)
  s = siteinds(ψ0)
  𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
  Uθ⃗ = ops(𝒰θ⃗, s)
  ψθ⃗ = apply(Uθ⃗, ψ0; cutoff=1e-8)
  return inner(ψθ⃗, H, ψθ⃗; cutoff=1e-8)
end

Random.seed!(1234)
θ⃗₀ = 2π * rand(nsites * nlayers)

@show loss(θ⃗₀)

println("\nOptimize circuit with gradient optimization")

loss_∇loss(x) = (loss(x), convert(Vector, loss'(x)))
algorithm = LBFGS(; gradtol=1e-3, verbosity=2)
θ⃗ₒₚₜ, lossₒₚₜ, ∇lossₒₚₜ, numfg, normgradhistory = optimize(loss_∇loss, θ⃗₀, algorithm)

@show loss(θ⃗ₒₚₜ)

println("\nRun DMRG as a comparison")

e_dmrg, ψ_dmrg = dmrg(H, ψ0; nsweeps=5, maxdim=10)

println("\nCompare variational circuit energy to DMRG energy")
@show loss(θ⃗ₒₚₜ), e_dmrg

nothing
