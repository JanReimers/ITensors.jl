using NDTensors
using LinearAlgebra
using Test

@testset "random_orthog" begin
  n, m = 10, 4
  O1 = random_orthog(n, m)
  @test eltype(O1) == Float64
  @test norm(transpose(O1) * O1 - Diagonal(fill(1.0, m))) < 1E-14
  O2 = random_orthog(m, n)
  @test norm(O2 * transpose(O2) - Diagonal(fill(1.0, m))) < 1E-14
end

@testset "random_unitary" begin
  n, m = 10, 4
  U1 = random_unitary(n, m)
  @test eltype(U1) == ComplexF64
  @test norm(U1' * U1 - Diagonal(fill(1.0, m))) < 1E-14
  U2 = random_unitary(m, n)
  @test norm(U2 * U2' - Diagonal(fill(1.0, m))) < 1E-14
end

@testset "Dense $qx decomposition, elt=$elt, positve=$positive" for qx in [qr, ql],
  elt in [Float64, ComplexF64, Float32, ComplexF32],
  positive in [false, true]

  eps = Base.eps(real(elt)) * 30 #this is set rather tight, so if you increase/change m,n you may have open up the tolerance on eps.
  n, m = 4, 8
  Id = Diagonal(fill(1.0, min(n, m)))
  #
  # Wide matrix (more columns than rows)
  #
  A = randomTensor(elt, (n, m))
  Q, X = qx(A; positive=positive) #X is R or L.
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Id atol = eps
  @test array(Q) * array(Q)' ≈ Id atol = eps
  if positive
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  #
  # Tall matrix (more rows than cols)
  #
  A = randomTensor(elt, (m, n)) #Tall array
  Q, X = qx(A; positive=positive)
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Id atol = eps
  if positive
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
end

@testset "Dense Rank revealing QR/RQ decomposition" begin
  n, m = 4, 8
  A = randomTensor(Float64,(n, m))
  # make some columns lineary dependent
  A[2, :] = A[1, :] * 1.1
  A[4, :] = A[1, :] * 2.1
  Q, R = qr(A; rr_cutoff=1e-12)
  @test dim(Q, 2) == n - 2 #make 2 columns actually got removed.
  @test dim(R, 1) == n - 2 #make 2 rows actually got removed.
  @test A ≈ Q * R atol = 1e-12
  nm = dim(Q, 2)
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, nm)) atol = 1e-12

  Q, L = ql(A; rr_cutoff=1e-12)
  @test dim(Q, 2) == n - 2 #make 2 rows actually got removed.
  @test dim(L, 1) == n - 2 #make 2 columns actually got removed.
  @test A ≈ Q * L atol = 1e-12
  nm = dim(Q, 2)
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, nm)) atol = 1e-12
end
nothing
