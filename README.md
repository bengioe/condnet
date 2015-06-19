# condnet

A policy driven sparse net, using explicit sparse dot products to go faster.  See [our RLDM submission](http://pierrelucbacon.com/bacon-2015-condnet.pdf).

## Sparse dot product

Please note that the custom Ops implemented here do **not** do any checking of shapes yet, so be careful.  
On CPU it is worth to do explicit sparse products for all three sparse/full matrix combinations.   
On GPU though, the cuBLAS GEMM operation is so fast that it is only worth doing a sparse matrix product when both matrices are sparse.

The expected input of the sparse_dot function are the matrices `A` and `B`, the (binary) mask `Am` by which `A` is multiplied, the mask `Om` by which the result of `dot(A,B)` is multiplied, and finally for convenience a bias vector `c` which is added (only where `Om` is nonzero), so the result is:  
`O = (dot(A*Am, B) + c) * Om`  
except that `Am` and `Om` are actually not the same shape as `A` and `O`. For a `(n,m)` matrix, the mask should be of shape `(n, m / block_size)`.

Also note that `A` is expected to already be 0 where `Am` is zero (this is in the GPU sparse-by-full case), which should already be the case if A is the output of a previous sparse operation.

### Performance

Typically the higher the block size and the higher the sparsity, the more speedup you will get. On CPU this seems to be less true for very large matrices, while on GPU it is the opposite (the speedup is large for large matrices). 

The CPU implementation is single-core (for now), but the intent is to have a proxy of how well it would perform on single-core hardware (e.g. a cheap phone).

### Stability

I'm still hunting down the bugs, as it seems that on the CPU (maybe for numerical precision reasons?) I don't exactly get the right results, at least for the gradients, because training with my Ops is slower, as if the gradient wasn't as precise. Or maybe my math is wrong.