
const BlockSparseMatrix{ElT,StoreT,IndsT} = BlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagBlockSparseMatrix{ElT,StoreT,IndsT} = DiagBlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagMatrix{ElT,StoreT,IndsT} = DiagTensor{ElT,2,StoreT,IndsT}

function _truncated_blockdim(
  S::DiagMatrix, docut::Real; singular_values=false, truncate=true, min_blockdim=0
)
  full_dim = diaglength(S)
  !truncate && return full_dim
  min_blockdim = min(min_blockdim, full_dim)
  newdim = 0
  val = singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
  while newdim + 1 ≤ full_dim && val > docut
    newdim += 1
    if newdim + 1 ≤ full_dim
      val =
        singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
    end
  end
  (newdim >= min_blockdim) || (newdim = min_blockdim)
  return newdim
end

"""
    svd(T::BlockSparseTensor{<:Number,2}; kwargs...)

svd of an order-2 BlockSparseTensor.

This function assumes that there is one block
per row/column, otherwise it fails.
This assumption makes it so the result can be
computed from the dense svds of seperate blocks.
"""
function LinearAlgebra.svd(T::BlockSparseMatrix{ElT}; kwargs...) where {ElT}
  alg::String = get(kwargs, :alg, "divide_and_conquer")
  min_blockdim::Int = get(kwargs, :min_blockdim, 0)
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  #@timeit_debug timer "block sparse svd" begin
  Us = Vector{DenseTensor{ElT,2}}(undef, nnzblocks(T))
  Ss = Vector{DiagTensor{real(ElT),2}}(undef, nnzblocks(T))
  Vs = Vector{DenseTensor{ElT,2}}(undef, nnzblocks(T))

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  for (n, b) in enumerate(eachnzblock(T))
    blockT = blockview(T, b)
    USVb = svd(blockT; alg=alg)
    if isnothing(USVb)
      return nothing
    end
    Ub, Sb, Vb = USVb
    Us[n] = Ub
    Ss[n] = Sb
    Vs[n] = Vb
    # Previously this was:
    # vector(diag(Sb))
    # But it broke, did `diag(::Tensor)` change types?
    # TODO: call this a function `diagonal`, i.e.:
    # https://github.com/JuliaLang/julia/issues/30250
    # or make `diag(::Tensor)` return a view by default.
    append!(d, data(Sb))
  end

  # Square the singular values to get
  # the eigenvalues
  d .= d .^ 2
  sort!(d; rev=true)

  # Get the list of blocks of T
  # that are not dropped
  nzblocksT = nzblocks(T)

  dropblocks = Int[]
  if truncate
    truncerr, docut = truncate!(d; kwargs...)
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(
        Ss[n], docut; min_blockdim, singular_values=true, truncate
      )
      if blockdim == 0
        push!(dropblocks, n)
      else
        Strunc = tensor(Diag(storage(Ss[n])[1:blockdim]), (blockdim, blockdim))
        Us[n] = Us[n][1:dim(Us[n], 1), 1:blockdim]
        Ss[n] = Strunc
        Vs[n] = Vs[n][1:dim(Vs[n], 1), 1:blockdim]
      end
    end
    deleteat!(Us, dropblocks)
    deleteat!(Ss, dropblocks)
    deleteat!(Vs, dropblocks)
    deleteat!(nzblocksT, dropblocks)
  else
    truncerr, docut = 0.0, 0.0
  end

  # The number of non-zero blocks of T remaining
  nnzblocksT = length(nzblocksT)

  #
  # Make indices of U and V 
  # that connect to S
  #
  i1 = ind(T, 1)
  i2 = ind(T, 2)
  uind = dag(sim(i1))
  vind = dag(sim(i2))
  resize!(uind, nnzblocksT)
  resize!(vind, nnzblocksT)
  for (n, blockT) in enumerate(nzblocksT)
    Udim = size(Us[n], 2)
    b1 = block(i1, blockT[1])
    setblock!(uind, resize(b1, Udim), n)
    Vdim = size(Vs[n], 2)
    b2 = block(i2, blockT[2])
    setblock!(vind, resize(b2, Vdim), n)
  end

  #
  # Put the blocks into U,S,V
  # 

  nzblocksU = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksS = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksV = Vector{Block{2}}(undef, nnzblocksT)

  for (n, blockT) in enumerate(nzblocksT)
    blockU = (blockT[1], UInt(n))
    nzblocksU[n] = blockU

    blockS = (n, n)
    nzblocksS[n] = blockS

    blockV = (blockT[2], UInt(n))
    nzblocksV[n] = blockV
  end

  indsU = setindex(inds(T), uind, 2)

  indsV = setindex(inds(T), vind, 1)
  indsV = permute(indsV, (2, 1))

  indsS = setindex(inds(T), dag(uind), 1)
  indsS = setindex(indsS, dag(vind), 2)

  U = BlockSparseTensor(ElT, undef, nzblocksU, indsU)
  S = DiagBlockSparseTensor(real(ElT), undef, nzblocksS, indsS)
  V = BlockSparseTensor(ElT, undef, nzblocksV, indsV)

  for n in 1:nnzblocksT
    Ub, Sb, Vb = Us[n], Ss[n], Vs[n]

    blockU = nzblocksU[n]
    blockS = nzblocksS[n]
    blockV = nzblocksV[n]

    if VERSION < v"1.5"
      # In v1.3 and v1.4 of Julia, Ub has
      # a very complicated view wrapper that
      # can't be handled efficiently
      Ub = copy(Ub)
      Vb = copy(Vb)
    end

    blockview(U, blockU) .= Ub
    blockviewS = blockview(S, blockS)
    for i in 1:diaglength(Sb)
      setdiagindex!(blockviewS, getdiagindex(Sb, i), i)
    end

    blockview(V, blockV) .= Vb
  end

  return U, S, V, Spectrum(d, truncerr)
  #end # @timeit_debug
end

_eigen_eltypes(T::Hermitian{ElT,<:BlockSparseMatrix{ElT}}) where {ElT} = real(ElT), ElT

_eigen_eltypes(T::BlockSparseMatrix{ElT}) where {ElT} = complex(ElT), complex(ElT)

function LinearAlgebra.eigen(
  T::Union{Hermitian{ElT,<:BlockSparseMatrix{ElT}},BlockSparseMatrix{ElT}}; kwargs...
) where {ElT<:Union{Real,Complex}}
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  ElD, ElV = _eigen_eltypes(T)

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  for b in eachnzblock(T)
    all(==(b[1]), b) || error("Eigen currently only supports block diagonal matrices.")
  end

  b = first(eachnzblock(T))
  blockT = blockview(T, b)
  Db, Vb = eigen(blockT)
  Ds = [Db]
  Vs = [Vb]
  append!(d, abs.(data(Db)))
  for (n, b) in enumerate(eachnzblock(T))
    n == 1 && continue
    blockT = blockview(T, b)
    Db, Vb = eigen(blockT)
    push!(Ds, Db)
    push!(Vs, Vb)
    append!(d, abs.(data(Db)))
  end

  dropblocks = Int[]
  sort!(d; rev=true, by=abs)

  if truncate
    truncerr, docut = truncate!(d; kwargs...)
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(Ds[n], docut)
      if blockdim == 0
        push!(dropblocks, n)
      else
        Dtrunc = tensor(Diag(storage(Ds[n])[1:blockdim]), (blockdim, blockdim))
        Ds[n] = Dtrunc
        Vs[n] = copy(Vs[n][1:dim(Vs[n], 1), 1:blockdim])
      end
    end
    deleteat!(Ds, dropblocks)
    deleteat!(Vs, dropblocks)
  else
    truncerr = 0.0
  end

  # Get the list of blocks of T
  # that are not dropped
  nzblocksT = nzblocks(T)
  deleteat!(nzblocksT, dropblocks)

  # The number of blocks of T remaining
  nnzblocksT = nnzblocks(T) - length(dropblocks)

  #
  # Put the blocks into D, V
  #

  i1, i2 = inds(T)
  l = sim(i1)

  lkeepblocks = Int[bT[1] for bT in nzblocksT]
  ldropblocks = setdiff(1:nblocks(l), lkeepblocks)
  deleteat!(l, ldropblocks)

  # l may have too many blocks
  (nblocks(l) > nnzblocksT) && error("New index l in eigen has too many blocks")

  # Truncation may have changed
  # some block sizes
  for n in 1:nnzblocksT
    setblockdim!(l, minimum(dims(Ds[n])), n)
  end

  r = dag(sim(l))

  indsD = (l, r)
  indsV = (dag(i2), r)

  nzblocksD = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksV = Vector{Block{2}}(undef, nnzblocksT)
  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockD = (n, n)
    nzblocksD[n] = blockD

    blockV = (blockT[1], n)
    nzblocksV[n] = blockV
  end

  D = DiagBlockSparseTensor(ElD, undef, nzblocksD, indsD)
  V = BlockSparseTensor(ElV, undef, nzblocksV, indsV)

  for n in 1:nnzblocksT
    Db, Vb = Ds[n], Vs[n]

    blockD = nzblocksD[n]
    blockviewD = blockview(D, blockD)
    for i in 1:diaglength(Db)
      setdiagindex!(blockviewD, getdiagindex(Db, i), i)
    end

    blockV = nzblocksV[n]
    blockview(V, blockV) .= Vb
  end

  return D, V, Spectrum(d, truncerr)
end

# QR a block sparse Rank 2 tensor.
#  This code thanks to Niklas Tausendpfund https://github.com/ntausend/variance_iTensor/blob/main/Hubig_variance_test.ipynb
#
function rq(T::BlockSparseTensor{ElT,2}; kwargs...) where {ElT}

  # getting total number of blocks
  nnzblocksT = nnzblocks(T)
  nzblocksT = nzblocks(T)

  Qs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)
  Rs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)

  for (jj, b) in enumerate(eachnzblock(T))
    blockT = blockview(T, b)
    RQb = rq(blockT; kwargs...) #call dense qr at src/linearalgebra.jl 387

    if (isnothing(RQb))
      return nothing
    end

    R, Q = RQb
    Qs[jj] = Q
    Rs[jj] = R
  end

  nb1_lt_nb2 = (
    nblocks(T)[1] < nblocks(T)[2] ||
    (nblocks(T)[1] == nblocks(T)[2] && dim(T, 1) < dim(T, 2))
  )

  # setting the left index of the Q isometry, this should be
  # the smaller index of the two indices of of T
  qindr = ind(T, 2)
  if nb1_lt_nb2
    qindl = sim(ind(T, 1))
  else
    qindl = sim(ind(T, 2))
  end

  # can qindl have more blocks than T?
  if nblocks(qindl) > nnzblocksT
    resize!(qindl, nnzblocksT)
  end

  for n in 1:nnzblocksT
    q_dim_red = minimum(dims(Rs[n]))
    NDTensors.setblockdim!(qindl, q_dim_red, n)
  end

  # correcting the direction of the arrow
  # if one have to be corrected the other one 
  # should also be corrected
  if (dir(qindl) != dir(qindr))
    qindl = dag(qindl)
  end

  indsQ = setindex(inds(T), dag(qindl), 1)
  indsR = setindex(inds(T), qindl, 2)

  nzblocksQ = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksR = Vector{Block{2}}(undef, nnzblocksT)

  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockR = (blockT[1], UInt(n))
    nzblocksR[n] = blockR

    blockQ = (UInt(n), blockT[2])
    nzblocksQ[n] = blockQ
  end

  Q = BlockSparseTensor(ElT, undef, nzblocksQ, indsQ)
  R = BlockSparseTensor(ElT, undef, nzblocksR, indsR)

  for n in 1:nnzblocksT
    Qb, Rb = Qs[n], Rs[n]
    blockQ = nzblocksQ[n]
    blockR = nzblocksR[n]

    if VERSION < v"1.5"
      # In v1.3 and v1.4 of Julia, Ub has
      # a very complicated view wrapper that
      # can't be handled efficiently
      Qb = copy(Qb)
      Rb = copy(Vb)
    end

    blockview(Q, blockQ) .= Qb
    blockview(R, blockR) .= Rb
  end

  return R, Q
end
# QR a block sparse Rank 2 tensor.
#  This code thanks to Niklas Tausendpfund https://github.com/ntausend/variance_iTensor/blob/main/Hubig_variance_test.ipynb
#
function LinearAlgebra.qr(T::BlockSparseTensor{ElT,2}; kwargs...) where {ElT}

  # getting total number of blocks
  nnzblocksT = nnzblocks(T)
  nzblocksT = nzblocks(T)

  Qs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)
  Rs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)

  for (jj, b) in enumerate(eachnzblock(T))
    blockT = blockview(T, b)
    QRb = qr(blockT; kwargs...) #call dense qr at src/linearalgebra.jl 387

    if (isnothing(QRb))
      return nothing
    end

    Q, R = QRb
    Qs[jj] = Q
    Rs[jj] = R
  end

  nb1_lt_nb2 = (
    nblocks(T)[1] < nblocks(T)[2] ||
    (nblocks(T)[1] == nblocks(T)[2] && dim(T, 1) < dim(T, 2))
  )

  # setting the right index of the Q isometry, this should be
  # the smaller index of the two indices of of T
  qindl = ind(T, 1)
  if nb1_lt_nb2
    qindr = sim(ind(T, 1))
  else
    qindr = sim(ind(T, 2))
  end

  # can qindr have more blocks than T?
  if nblocks(qindr) > nnzblocksT
    resize!(qindr, nnzblocksT)
  end

  for n in 1:nnzblocksT
    q_dim_red = minimum(dims(Rs[n]))
    NDTensors.setblockdim!(qindr, q_dim_red, n)
  end

  # correcting the direction of the arrow
  # since qind2r is basically a copy of qind1r
  # if one have to be corrected the other one 
  # should also be corrected
  if (dir(qindr) != dir(qindl))
    qindr = dag(qindr)
  end

  indsQ = setindex(inds(T), dag(qindr), 2)
  indsR = setindex(inds(T), qindr, 1)

  nzblocksQ = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksR = Vector{Block{2}}(undef, nnzblocksT)

  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockQ = (blockT[1], UInt(n))
    nzblocksQ[n] = blockQ

    blockR = (UInt(n), blockT[2])
    nzblocksR[n] = blockR
  end

  Q = BlockSparseTensor(ElT, undef, nzblocksQ, indsQ)
  R = BlockSparseTensor(ElT, undef, nzblocksR, indsR)

  for n in 1:nnzblocksT
    Qb, Rb = Qs[n], Rs[n]
    blockQ = nzblocksQ[n]
    blockR = nzblocksR[n]

    if VERSION < v"1.5"
      # In v1.3 and v1.4 of Julia, Ub has
      # a very complicated view wrapper that
      # can't be handled efficiently
      Qb = copy(Qb)
      Rb = copy(Vb)
    end

    blockview(Q, blockQ) .= Qb
    blockview(R, blockR) .= Rb
  end

  return Q, R
end

function exp(
  T::Union{BlockSparseMatrix{ElT},Hermitian{ElT,<:BlockSparseMatrix{ElT}}}
) where {ElT<:Union{Real,Complex}}
  expT = BlockSparseTensor(ElT, undef, nzblocks(T), inds(T))
  for b in eachnzblock(T)
    all(==(b[1]), b) || error("exp currently supports only block-diagonal matrices")
  end
  for b in eachdiagblock(T)
    blockT = blockview(T, b)
    if isnothing(blockT)
      # Block was not found in the list, treat as 0
      id_block = Matrix{ElT}(I, blockdims(T, b))
      insertblock!(expT, b)
      blockview(expT, b) .= id_block
    else
      blockview(expT, b) .= exp(blockT)
    end
  end
  return expT
end
