export eigs, entropy, polar, random_orthog, random_unitary, Spectrum, svd, truncerror

#
# Linear Algebra of order 2 NDTensors
#
# Even though DenseTensor{_,2} is strided
# and passable to BLAS/LAPACK, it cannot
# be made <: StridedArray

function (
  T1::Tensor{ElT1,2,StoreT1} * T2::Tensor{ElT2,2,StoreT2}
) where {ElT1,StoreT1<:Dense,ElT2,StoreT2<:Dense}
  RM = matrix(T1) * matrix(T2)
  indsR = (ind(T1, 1), ind(T2, 2))
  return tensor(Dense(vec(RM)), indsR)
end

function LinearAlgebra.exp(T::DenseTensor{ElT,2}) where {ElT<:Union{Real,Complex}}
  expTM = exp(matrix(T))
  return tensor(Dense(vec(expTM)), inds(T))
end

function LinearAlgebra.exp(
  T::Hermitian{ElT,<:DenseTensor{ElT,2}}
) where {ElT<:Union{Real,Complex}}
  # exp(::Hermitian/Symmetric) returns Hermitian/Symmetric,
  # so extract the parent matrix
  expTM = parent(exp(matrix(T)))
  return tensor(Dense(vec(expTM)), inds(T))
end

"""
  Spectrum
contains the (truncated) density matrix eigenvalue spectrum which is computed during a
decomposition done by `svd` or `eigen`. In addition stores the truncation error.
"""
struct Spectrum{VecT<:Union{AbstractVector,Nothing},ElT<:Real}
  eigs::VecT
  truncerr::ElT
end

eigs(s::Spectrum) = s.eigs
truncerror(s::Spectrum) = s.truncerr

function entropy(s::Spectrum)
  S = 0.0
  eigs_s = eigs(s)
  isnothing(eigs_s) &&
    error("Spectrum does not contain any eigenvalues, cannot compute the entropy")
  for p in eigs_s
    p > 1e-13 && (S -= p * log(p))
  end
  return S
end

function svd_catch_error(A; kwargs...)
  USV = try
    svd(A; kwargs...)
  catch
    return nothing
  end
  return USV
end

function lapack_svd_error_message(alg)
  return "The SVD algorithm `\"$alg\"` has thrown an error,\n" *
         "likely because of a convergance failure. You can try\n" *
         "other SVD algorithms that may converge better using the\n" *
         "`alg` (or `svd_alg` if called through `factorize` or MPS/MPO functionality) keyword argument:\n\n" *
         " - \"divide_and_conquer\" is a divide-and-conquer algorithm\n" *
         "   (LAPACK's `gesdd`). It is fast, but may lead to some innacurate\n" *
         "   singular values for very ill-conditioned matrices.\n" *
         "   It also may sometimes fail to converge, leading to errors\n" *
         "   (in which case `\"qr_iteration\"` or `\"recursive\"` can be tried).\n\n" *
         " - `\"qr_iteration\"` (LAPACK's `gesvd`) is typically slower \n" *
         "   than \"divide_and_conquer\", especially for large matrices,\n" *
         "   but is more accurate for very ill-conditioned matrices \n" *
         "   compared to `\"divide_and_conquer\"`.\n\n" *
         " - `\"recursive\"` is ITensor's custom SVD algorithm. It is very\n" *
         "   reliable, but may be slow if high precision is needed.\n" *
         "   To get an `svd` of a matrix `A`, an eigendecomposition of\n" *
         "   ``A^{\\dagger} A`` is used to compute `U` and then a `qr` of\n" *
         "   ``A^{\\dagger} U`` is used to compute `V`. This is performed\n" *
         "   recursively to compute small singular values.\n\n" *
         "Returning `nothing`. For an output `F = svd(A, ...)` you can check if\n" *
         "`isnothing(F)` in your code and try a different algorithm.\n\n" *
         "To suppress this message in the future, you can wrap the `svd` call in the\n" *
         "`@suppress` macro from the `Suppressor` package.\n"
end

"""
    svd(T::DenseTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function LinearAlgebra.svd(T::DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  #
  # Keyword argument deprecations
  #
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end

  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  if haskey(kwargs, :fastsvd) || haskey(kwargs, :fastSVD)
    error(
      "In svd, fastsvd/fastSVD keyword arguments are removed in favor of alg, see documentation for more details.",
    )
  end

  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)
  alg::String = get(kwargs, :alg, "divide_and_conquer")

  #@timeit_debug timer "dense svd" begin
  if alg == "divide_and_conquer"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.DivideAndConquer())
    if isnothing(MUSV)
      # If "divide_and_conquer" fails, try "qr_iteration"
      alg = "qr_iteration"
      MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
      if isnothing(MUSV)
        # If "qr_iteration" fails, try "recursive"
        alg = "recursive"
        MUSV = svd_recursive(matrix(T))
      end
    end
  elseif alg == "qr_iteration"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
    if isnothing(MUSV)
      # If "qr_iteration" fails, try "recursive"
      alg = "recursive"
      MUSV = svd_recursive(matrix(T))
    end
  elseif alg == "recursive"
    MUSV = svd_recursive(matrix(T))
  else
    error(
      "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
    )
  end
  if isnothing(MUSV)
    if any(isnan, T)
      println("SVD failed, the matrix you were trying to SVD contains NaNs.")
    else
      println(lapack_svd_error_message(alg))
    end
    return nothing
  end
  MU, MS, MV = MUSV
  conj!(MV)
  #end # @timeit_debug

  P = MS .^ 2
  if truncate
    truncerr, _ = truncate!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:, 1:dS]
    resize!(MS, dS)
    MV = MV[:, 1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T, 1), u))
  Sinds = IndsT((u, v))
  Vinds = IndsT((ind(T, 2), v))
  U = tensor(Dense(vec(MU)), Uinds)
  S = tensor(Diag(MS), Sinds)
  V = tensor(Dense(vec(MV)), Vinds)
  return U, S, V, spec
end

function LinearAlgebra.eigen(
  T::Hermitian{ElT,<:DenseTensor{ElT,2,IndsT}}; kwargs...
) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff::Union{Nothing,Float64} = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(matrixT)

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; rev=true, by=abs)
  DM = DM[p]
  VM = VM[:, p]

  if truncate
    truncerr, _ = truncate!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(DM, truncerr)

  # Make the new indices to go onto V
  l = eltype(IndsT)(dD)
  r = eltype(IndsT)(dD)
  Vinds = IndsT((dag(ind(T, 2)), dag(r)))
  Dinds = IndsT((l, dag(r)))
  V = tensor(Dense(vec(VM)), Vinds)
  D = tensor(Diag(DM), Dinds)
  return D, V, spec
end

"""
    random_unitary(n::Int,m::Int)::Matrix{ComplexF64}
    random_unitary(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random matrix U of dimensions (n,m)
such that if n >= m, U'*U is the identity, or if
m > n U*U' is the identity. Optionally can pass a numeric
type as the first argument to obtain a matrix of that type.

Sampling is based on https://arxiv.org/abs/math-ph/0609050
such that in the case `n==m`, the unitary matrix will be sampled
according to the Haar measure.
"""
function random_unitary(::Type{ElT}, n::Int, m::Int) where {ElT<:Number}
  if n < m
    return Matrix(random_unitary(ElT, m, n)')
  end
  F = qr(randn(ElT, n, m))
  Q = Matrix(F.Q)
  # The upper triangle of F.factors 
  # are the elements of R.
  # Multiply cols of Q by the signs
  # that would make diagonal of R 
  # non-negative:
  for c in 1:size(Q, 2)
    Q[:, c] .*= sign(F.factors[c, c])
  end
  return Q
end

random_unitary(n::Int, m::Int) = random_unitary(ComplexF64, n, m)

"""
    random_orthog(n::Int,m::Int)::Matrix{Float64}
    random_orthog(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random, real matrix O of dimensions (n,m)
such that if n >= m, transpose(O)*O is the
identity, or if m > n O*transpose(O) is the
identity. Optionally can pass a real number type
as the first argument to obtain a matrix of that type.
"""
random_orthog(::Type{ElT}, n::Int, m::Int) where {ElT<:Real} = random_unitary(ElT, n, m)

random_orthog(n::Int, m::Int) = random_orthog(Float64, n, m)

"""
    qr_positive(M::AbstractMatrix)

Compute the QR decomposition of a matrix M
such that the diagonal elements of R are
non-negative. Such a QR decomposition of a
matrix is unique. Returns a tuple (Q,R).
"""
function qr_positive(M::AbstractMatrix)
  sparseQ, R = qr(M)
  Q = convert(Matrix, sparseQ)
  nc = size(Q, 2)
  for c in 1:nc
    if real(R[c, c]) < 0.0
      R[c, c:end] *= -1
      Q[:, c] *= -1
    end
  end
  return (Q, R)
end

function LinearAlgebra.eigen(
  T::DenseTensor{ElT,2,IndsT}; kwargs...
) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff::Float64 = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(matrixT)

  # Sort by largest to smallest eigenvalues
  #p = sortperm(DM; rev = true)
  #DM = DM[p]
  #VM = VM[:,p]

  if truncate
    truncerr, _ = truncate!(
      DM; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(abs.(DM), truncerr)

  i1, i2 = inds(T)

  # Make the new indices to go onto D and V
  l = typeof(i1)(dD)
  r = dag(sim(l))
  Dinds = (l, r)
  Vinds = (dag(i2), r)
  D = complex(tensor(Diag(DM), Dinds))
  V = complex(tensor(Dense(vec(VM)), Vinds))
  return D, V, spec
end
#
#  QR rank reduction helpers
#
function find_zero_rows(R::AbstractMatrix,eps::Float64)::Array{Bool}  where {ElT,IndsT}
  nr,nc=size(R)
  zeros=falses(nr)
  for r in 1:nr
    s=0.0
    for c in 1:nc
      s=max(s,abs(R[r, c]))
    end
    zeros[r]= (s<=eps)
  end
  return zeros
end

#
#  Trim out zero rows of R within tolerance eps. Also trim the corresponding columns
#  of Q.
#
function trim_rows(R::AbstractMatrix,Q::AbstractMatrix,eps::Float64) where {ElT,IndsT}
  zeros=find_zero_rows(R,eps)
  num_zero_rows=sum(zeros)
  if num_zero_rows==0
    return R,Q
  end
  #@printf "Rank Reveal removing %4i rows with epsrr=%.1e\n" num_zero_rows eps
  Rnr,Rnc=size(R)
  Qnr,Qnc=size(Q)
  #@assert Rnr==Qnc Q is strided to we can't asume this
  R1nr=Rnr-num_zero_rows
  T=eltype(R)
  R1=Matrix{T}(undef,R1nr,Rnc)
  Q1=Matrix{T}(undef,Qnr ,R1nr)
  r1=1
  for r in 1:Rnr
    if zeros[r]==false
      R1[r1,:]=R[r,:] #transfer row
      Q1[:,r1]=Q[:,r] #transfer column
      r1+=1 #next row in rank reduces matrices.
    end #if zero
  end #for r
  return R1,Q1
end
#
#  Trim out zero columnss of R within tolerance eps. Also trim the corresponding rows
#  of Q.
#
function trim_columns(R::AbstractMatrix,Q::AbstractMatrix,eps::Float64) where {ElT,IndsT}
  R,Q=trim_rows(transpose(R),transpose(Q),eps) 
  return transpose(R),transpose(Q) 
end

function LinearAlgebra.qr(T::DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  positive = get(kwargs, :positive, false)
  # TODO: just call qr on T directly (make sure
  # that is fast)
  if positive
    QM, RM = qr_positive(matrix(T))
  else
    QM, RM = qr(matrix(T))
  end
  #
  #  Do row removal for rank revealing RQ
  #
  epsrr::Float64 = get(kwargs, :epsrr , -1.0)
  if epsrr>=0.0 
    RM,QM=trim_rows(RM,QM,epsrr) 
  end
  #
  # Make the new indices to go onto Q and R
  #
  IndexT=IndsT.parameters[1]
  nq = IndexT(size(RM)[1]) #dim of the link index
  Qinds = IndsT((ind(T, 1), nq))
  Rinds = IndsT((nq, ind(T, 2)))
  Q = tensor(Dense(vec(Matrix(QM))), Qinds) #Q was strided
  R = tensor(Dense(vec(RM)), Rinds)
  return Q, R
end

#
#  Uses kwargs:positive to decide which rq method to call.
#
function rq(T::DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  if get(kwargs, :positive, false)
    RM, QM = rq_positive(matrix(T))
  else
    RM, QM = rq(matrix(T))
  end
  
  
  #
  #  Do row removal for rank revealing RQ
  #
  epsrr::Float64 = get(kwargs, :epsrr , -1.0)
  if epsrr>=0.0 
    RM,QM=trim_columns(RM,QM,epsrr) 
  end
  #
  # Make the new indices to go onto Q and R
  #
  IndexT=IndsT.parameters[1]
  nq = IndexT(size(RM)[2]) #dim of the link index
  Qinds = IndsT((nq,ind(T, 2)))
  Rinds = IndsT((ind(T, 1),nq))
  Q = NDTensors.tensor(NDTensors.Dense(vec(Matrix(QM))), Qinds) #Q was strided
  R = NDTensors.tensor(NDTensors.Dense(vec(RM)), Rinds)
  return R, Q
end

#
# Just flip signs between Q and R to get all the diagonals of R >=0.
# For rectangular M the indexing for "diagonal" is non-trivial.
#
function rq_positive(M::AbstractMatrix)
  R, sparseQ = rq(M)
  Q = convert(Matrix, sparseQ)
  nr, nc = size(R)
  dr = nr > nc ? nr - nc : 0 #diag is shifted down by dr if nr>nc
  for r in 1:nr
    if r <= nc && real(R[r + dr, r]) < 0.0
      R[1:(r + dr), r] *= -1
      Q[r, :] *= -1
    end
  end
  return (R, Q)
end

#
#  Lapack replaces A with Q & R carefully packed together.  So here we just copy a
#  before letting lapack overwirte it. 
#
function rq(A::AbstractMatrix{T}; kwargs...) where {T}
  Base.require_one_based_indexing(A)
  AA = similar(A, LinearAlgebra._qreltype(T), size(A))
  copyto!(AA, A)
  return rq!(AA; kwargs...)
end

rq!(A::AbstractMatrix) = rq!(A)

#
# This is where the low level call to lapack actually occurs.  Most of the work is
# about unpacking Q and R from the A matrix.
#
function rq!(A::StridedMatrix{<:LAPACK.BlasFloat})
  tau = similar(A, Base.min(size(A)...))
  x = LAPACK.gerqf!(A, tau)

  # Unpack R from the lower portion of A, before orgql! mangles it!
  nr, nc = size(A)
  mn = Base.min(nr, nc)
  R = similar(A, (nr, mn))
  for c in 1:mn
    for r in 1:(c + nr - mn)
      R[r, c] = A[r, c + nc - mn]
    end
    for r in (c + 1 + nr - mn):nr
      R[r, c] = 0.0
    end
  end
  #
  # If nr>nc we need shift the orth vectors from the bottom of Q up to top before
  # unpacking the reflectors.
  #
  if mn < nr
    for c in 1:nc
      for r in 1:mn
        A[r, c] = A[r + nr - mn, c]
      end
    end
    A = A[1:mn, :] #whack the extra rows in A or orgrq! will complain
  end
  LAPACK.orgrq!(A, tau)
  return R, A
end

# TODO: support alg keyword argument to choose the svd algorithm
function polar(T::DenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM, RM = polar(matrix(T))
  dim = size(QM, 2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  # TODO: use push/pushfirst instead of a constructor
  # call here
  Qinds = IndsT((ind(T, 1), q))
  Rinds = IndsT((q, ind(T, 2)))
  Q = tensor(Dense(vec(QM)), Qinds)
  R = tensor(Dense(vec(RM)), Rinds)
  return Q, R
end
