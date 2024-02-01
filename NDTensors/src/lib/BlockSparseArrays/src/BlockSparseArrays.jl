module BlockSparseArrays
include("blocksparsearrayinterface/blocksparsearrayinterface.jl")
include("blocksparsearrayinterface/linearalgebra.jl")
include("blocksparsearrayinterface/blockzero.jl")
include("blocksparsearrayinterface/broadcast.jl")
include("abstractblocksparsearray/abstractblocksparsearray.jl")
include("abstractblocksparsearray/wrappedabstractblocksparsearray.jl")
include("abstractblocksparsearray/abstractblocksparsematrix.jl")
include("abstractblocksparsearray/abstractblocksparsevector.jl")
include("abstractblocksparsearray/arraylayouts.jl")
include("abstractblocksparsearray/sparsearrayinterface.jl")
include("abstractblocksparsearray/linearalgebra.jl")
include("abstractblocksparsearray/broadcast.jl")
include("abstractblocksparsearray/map.jl")
include("blocksparsearray/defaults.jl")
include("blocksparsearray/blocksparsearray.jl")
include("BlockArraysExtensions/BlockArraysExtensions.jl")
include("BlockArraysSparseArrayInterfaceExt/BlockArraysSparseArrayInterfaceExt.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/src/BlockSparseArraysTensorAlgebraExt.jl")
end