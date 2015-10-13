
import theano
import theano.tensor as T
import numpy
import time
from theano.sandbox.cuda.basic_ops import gpu_from_host

def safe_to_gpu(x):
    if isinstance(x.type, T.TensorType):
        return gpu_from_host(x)
    else:
        return x





class DualMatrixNonzero(theano.sandbox.cuda.GpuOp):
    """
    This OP returns two things:
      - the equivalent of T.stack(*A.nonzero()), D
      - a coded nonzero index matrix C

    C is defined as such:
    [ [ start position of non-zeros in D for row 0, number of non-zeros for row 0]
      ...
      ...]
    
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "GPUMatrixNonzero"

    def make_node(self, A):
        assert A.ndim == 2
        A = safe_to_gpu(A)
        return theano.Apply(self,
                            inputs=[A],
                            outputs=[A.type(),#T.TensorType('int32', [False, False])(), 
                                     A.type()])#T.TensorType('int32', [False, False])()])
    
    def c_support_code(self, *args, **kwargs):
        return """int __warned_transposed = 0;"""
    def c_code_cache_version(self):
        return (2,3)

    def c_code(self, node, name, inp, out, sub):
        A, = inp
        B, C = out
        fail = sub['fail']
        return """

cudaError_t sts;
float * gpu_A = CudaNdarray_DEV_DATA(%(A)s);
float * gpu_B;
float * gpu_C;
const int * A_dims = CudaNdarray_HOST_DIMS(%(A)s);
const int * A_strides = CudaNdarray_HOST_STRIDES(%(A)s);
int A_size = A_dims[0] * A_dims[1];
float * cpu_A = (float*)malloc(sizeof(float) * A_size); 
cudaMemcpy(cpu_A, gpu_A, sizeof(float) * A_size, cudaMemcpyDeviceToHost);

int n_nonzeros = 0;
int* nonzero_positions = (int*)malloc(A_size * 2 * sizeof(int));
int* nzp_yptr = nonzero_positions;
int* nzp_xptr = nonzero_positions + A_size;


//printf("%%d %%d %%d %%d\\n",A_strides[0],A_strides[1],A_dims[0], A_dims[1]);
if (__warned_transposed == 0 && A_strides[0] != A_dims[1]){
    printf("Warning: it seems you're trying to run GpuNonzero on a transposed matrix, with the current implementation this is slow!\\n");
   __warned_transposed = 1;
}

int A_strides_1 = A_strides[1];
if (A_strides_1 == 0) A_strides_1++;

float *Aptr = cpu_A;
for (int i=0; i<A_size; i++){
  //printf("%%d %%d %%d %%d %%d\\n",i, A_strides[0], A_strides[1], A_dims[0], A_dims[1]);
  if (*Aptr++ != 0.0){
    *nzp_yptr++ = (i / A_strides[0]) %% A_dims[0];
    *nzp_xptr++ = (i / A_strides_1 ) %% A_dims[1];
    n_nonzeros++;
  }
}

int* nonzero_positions_compressed = (int*)malloc(n_nonzeros * 2 * sizeof(int));
memcpy(nonzero_positions_compressed, nonzero_positions,
       n_nonzeros * sizeof(int));
memcpy(nonzero_positions_compressed + n_nonzeros, nonzero_positions + A_size,
       n_nonzeros * sizeof(int));
free(cpu_A);
free(nonzero_positions);

int B_dims[] = {2, n_nonzeros};
int C_dims[] = {A_dims[0], 2};

//printf("asd\\n");

int* row_indexes = (int*)malloc(A_dims[0] * 2 * sizeof(int));
memset(row_indexes, 0, A_dims[0] * 2 * sizeof(int));
int last_start = 0;
int last_row = 0;
int count = 0;
for (int i=0; i<n_nonzeros; i++){
  int k = nonzero_positions_compressed[i];
  if (k != last_row){
    row_indexes[last_row * 2] = last_start;
    row_indexes[last_row * 2 + 1] = count;
    last_start = i;
    count = 0;
    last_row = k;
  }
  count++;
}
row_indexes[last_row * 2] = last_start;
row_indexes[last_row * 2 + 1] = count;

if (CudaNdarray_prep_output(&%(B)s, 2, B_dims)){
  %(fail)s;
 }

gpu_B = CudaNdarray_DEV_DATA(%(B)s);
cudaMemcpy(gpu_B, nonzero_positions_compressed, sizeof(int) * 2 * n_nonzeros, 
           cudaMemcpyHostToDevice);
free(nonzero_positions_compressed);

if (CudaNdarray_prep_output(&%(C)s, 2, C_dims)){
  %(fail)s;
 }

gpu_C = CudaNdarray_DEV_DATA(%(C)s);
cudaMemcpy(gpu_C, row_indexes, sizeof(int) * 2 * A_dims[0], 
           cudaMemcpyHostToDevice);
free(row_indexes);

sts = cudaGetLastError();
if (cudaSuccess != sts){
  PyErr_Format(PyExc_RuntimeError,
	       "Cuda error: gpunonzero: %%s",
	       cudaGetErrorString(sts));
            %(fail)s;
 }


        """ % locals()






class _GPUSparseDot_SparseBySparse(theano.sandbox.cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other) and self.block_size == other.block_size

    def __hash__(self):
        return hash(type(self))+hash(self.block_size)

    def __init__(self, block_size):
        self.block_size = block_size

    def make_node(self, A, Amask, B, Omask, c):
        if Amask.dtype!='float32':
            Amask = T.cast(Amask, 'float32')
        if Omask.dtype!='float32':
            Omask = T.cast(Omask, 'float32')

        aindexes,aindexes_data = DualMatrixNonzero()(Amask)
        indexes,indexes_data = DualMatrixNonzero()(Omask)            
            
        if indexes.dtype!='float32':
            print "?"
            indexes = T.cast(indexes, 'float32')
        if A.dtype!='float32':
            A = T.cast(A, 'float32')
        if B.dtype!='float32':
            B = T.cast(B, 'float32')
        A,B,c = [safe_to_gpu(i) for i in [A,B,c]]
        n,m,p = [T.as_tensor_variable(i) for i in [A.shape[0],B.shape[0],B.shape[1]]]

        return theano.Apply(self,
                            inputs=[A,aindexes,aindexes_data, 
                                    B,indexes,indexes_data, 
                                    n,m,p,Omask,c,Amask],
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,0))()])


    def grad(self, inputs, outputs):
        a,ia,iadata,b,i,idata,n,m,p,Omask,c,Amask = inputs
        gz, = outputs
        
        # most of these grads can be rewritten is sparse_dot-s, but I
        # need to make sure it's gonna work with transposed inputs
        # first (include strides), and also a zero bias (otherwise
        # it's also a waste of computation)
        gzo = (gz.reshape((gz.shape[0], Omask.shape[1], -1)) * Omask.dimshuffle(0,1,'x')).reshape(gz.shape)
        xgrad = T.dot(gzo, b.T)
        xgrad = (xgrad.reshape((a.shape[0], Amask.shape[1], -1)) * Amask.dimshuffle(0,1,'x')).reshape(xgrad.shape)
        a_masked = (a.reshape((a.shape[0], Amask.shape[1], -1)) * Amask.dimshuffle(0,1,'x')).reshape(a.shape)
        ygrad = T.dot(a_masked.T, gzo)
        cgrad = gzo.sum(axis=0)

        # MY CONCLUSION is that the gradient wrt to the Omask is
        # defined, but is zero since the Omask matrix should really
        # be of type "boolean", which is like a step function (0 derivative)
        dgrad = T.zeros_like(T.cast(Omask,'float32'))
        disc = theano.gradient.DisconnectedType()
        return tuple([xgrad,disc(),disc(),ygrad] + 
                     [disc() for i in [i,idata,n,m,p]]+
                     [dgrad, cgrad, disc()])
    def connection_pattern(self, node): 
        # a, aidx, aidxdata
        # b, bidx, bidxdata
        # n, m, p, Omask
        return [[True], [False], [False],
                [True], [False], [False],
                [False], [False], [False], [True], [True], [False]]

    def c_support_code(self):
        return """

__global__ void sparsedot_ss(float* A, float* B, int* indexes, int* aindexes, int* aindexes_data,
        float* C, int n, int m, int p, 
        float* bias,
        int n_indexes, int n_aindexes,
        int a_stride0, int a_stride1, int b_stride0, int b_stride1) {
        /*
        A is n,m 
        B is m,p 
        indexes is 2,n_indexes
        aindexes_data is n,2 = [...,[start_index, number_of_blocks],...]
        C is n,p
        */
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Abuf0[BLOCK_SIZE];
    int index_position = blockIdx.x * NTHREADS + threadIdx.x;
    if (index_position >= n_indexes) return;
    int tx = threadIdx.y;
    // this is the position of the computed value in C
    int posy = indexes[index_position]; // this is actually the minibatch index
    int posx = indexes[index_position + n_indexes] * BLOCK_SIZE + threadIdx.y;
    //if (index_position == 0 && tx ==0) { printf("%%d -- %%d %%d\\n",tx, posy, posx);}

    // this is where the list of positions in aindexes start (there are number_of_blocks positions)
    const int positions_start = aindexes_data[posy * 2] + n_aindexes;
    // this is the number of blocks that are non-zero in A
    const int number_of_blocks = aindexes_data[posy * 2 + 1];
    //printf("pos %%d: n %%d start %%d\\n", posy, number_of_blocks, positions_start - n_indexes);
    double acc0 = bias[posx];
    for (int ks=0;ks<number_of_blocks;ks++){
        //printf("%%d %%d %%d %%d\\n",n_indexes, aindexes[positions_start+ks], positions_start, ks);
        int aposx = aindexes[positions_start + ks] * BLOCK_SIZE;
        //if (index_position == 0 && tx==0) { printf("%%d aposx\\n",aposx);} 
        Abuf0[tx] = A[posy * a_stride0 + (tx + aposx) * a_stride1];
        __syncthreads();
        #pragma unroll BLOCK_SIZE
        for (int k=0;k<BLOCK_SIZE;++k){
            //if (k >= m) { break;} // this will never happen now, fix?
            double a = Abuf0[k];
            double b = B[((k+aposx) * b_stride0) + (posx * b_stride1)];
            //if (index_position == 0 && tx==0) { printf("ks: %%d aposx: %%d, A[%%d %%d]=%%f, B[%%d %%d] = %%f\\n",ks, aposx, posy, aposx+k,a,k+aposx,posx,b);} 
            acc0 += a * b;
        }
        __syncthreads();
    }
    C[posy * p + posx] = acc0;
    
}""" % (self.block_size)


    def c_code(self, node, name, inp, out, sub):
        A,aindexes,aindexes_data,B,indexes,indexes_data,n,m,p,Omask,c,Amask = inp
        z, = out
        fail = sub['fail']
        s = """
        float *A_data = CudaNdarray_DEV_DATA(%(A)s);
        float *B_data = CudaNdarray_DEV_DATA(%(B)s);
        int *I_data   = (int*)CudaNdarray_DEV_DATA(%(indexes)s);
        int *AI_data  = (int*)CudaNdarray_DEV_DATA(%(aindexes)s);
        int *AID_data = (int*)CudaNdarray_DEV_DATA(%(aindexes_data)s);
        float* bias_data = CudaNdarray_DEV_DATA(%(c)s);
        int dims[] = {0,0,0};
        float *O_data; // output data

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        dims[2] = ((dtype_%(p)s*)PyArray_DATA(%(p)s))[0];
        int total_size = dims[0] * dims[2] * sizeof(float);
        int zdims[] = {0,0};
        zdims[0] = dims[0];
        zdims[1] = dims[2];

        const int* index_dims = CudaNdarray_HOST_DIMS(%(indexes)s);
        const int* aindex_dims = CudaNdarray_HOST_DIMS(%(aindexes)s);
        const int* A_strides = CudaNdarray_HOST_STRIDES(%(A)s);
        const int* B_strides = CudaNdarray_HOST_STRIDES(%(B)s);

        int grid_size = index_dims[1] / NTHREADS;
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless excatly a fit (modulo is zero)
        if (index_dims[1] %% (NTHREADS) != 0) grid_size++;

        cudaError_t sts;
        void * orig_z = %(z)s;
        if (CudaNdarray_prep_output(&%(z)s, 2, zdims))
        {
            %(fail)s;
        }
        //printf("0-fill %%d\\n",total_size);
        sts = cudaMemset(CudaNdarray_DEV_DATA(%(z)s), 0, total_size);
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_MemoryError,
                         "GpuEye: Error in memset %%d bytes of device memory.",
                         total_size);
            if(orig_z == NULL)
                Py_XDECREF(%(z)s);
            %(fail)s;
        }
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        O_data = CudaNdarray_DEV_DATA(%(z)s);

        if (grid_size > 0){
        sparsedot_ss<<<grid_size, blocks>>>(A_data, B_data, I_data, AI_data, AID_data, O_data,
                                         dims[0], dims[1], dims[2], bias_data,
                                         index_dims[1], aindex_dims[1],
                                         A_strides[0], A_strides[1], B_strides[0], B_strides[1]); 
        }
        CNDA_THREAD_SYNC;

        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: sparse_dot: %%s. n=%%d, m=%%d, p=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), bstrides=(%%d %%d)",
                    cudaGetErrorString(sts),
                    dims[0], dims[1], dims[2], grid_size, blocks.x, blocks.y,
                    index_dims[0], index_dims[1],
                    B_strides[0], B_strides[1]);
            %(fail)s;
        }
        """ % locals()
        return s

    def c_code_cache_version(self):
        return (2,self.block_size)

    def __str__(self):
        return "GPUSparseDot_SS"









######################################################################
#
#
# The rest of the ops here aren't actually very efficient, so I would
# advise against using them in most cases
#
#
#





class _GPUSparseDot_FullBySparse(theano.sandbox.cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, *args):
        if len(args) == 4:
            A,B,dropout,block_size = args
            indexes,indexes_data = DualMatrixNonzero()(dropout)#T.stack(*dropout.nonzero())
        else:
            A,B,dropout,_,block_size = args
            indexes, indexes_data = _
            
        self.block_size = block_size
        if indexes.dtype!='float32':
            indexes = T.cast(indexes, 'float32')
        if A.dtype!='float32':
            A = T.cast(A, 'float32')
        if B.dtype!='float32':
            B = T.cast(B, 'float32')
        A,B,indexes = [safe_to_gpu(i) for i in [A,B,indexes]]
        n,m,p = [T.as_tensor_variable(i) for i in [A.shape[0],B.shape[0],B.shape[1]]]
        return theano.Apply(self,
                            inputs=[A,B,indexes, indexes_data, n,m,p,dropout],
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,0))(),
                                     indexes.type(),
                                     indexes_data.type()])


    def grad(self, inputs, outputs):
        a,b,i,idata,n,m,p,dropout = inputs
        gz,ai,aid = outputs
        
        gzo = (gz.reshape((gz.shape[0], dropout.shape[1], -1)) * dropout.dimshuffle(0,1,'x')).reshape(gz.shape)
        xgrad = T.dot(gzo, b.T)
        ygrad = T.dot(a.T, gzo)

        # MY CONCLUSION is that the gradient wrt to the dropout is
        # defined, but is zero since the dropout matrix should really
        # be of type "boolean", which is like a step function (0 derivative)
        dgrad = T.zeros_like(T.cast(dropout,'float32'))
        return tuple([xgrad,ygrad] + 
                     [theano.gradient.DisconnectedType()() for i in [i,idata,n,m,p]]+
                     [dgrad])
    def connection_pattern(self, node): 
        return [[True,False,False], [True,False,False], [False,False,False], 
                [False,False,False], [False,False,False], [False,False,False], 
                [False,False,False], [True,False,False]]

    def c_support_code(self):
        return """

__global__ void sparsedot_fs(float* X, float* W, float* indexes, float* z, int n, int m, int p, 
        int n_indexes,
        int x_stride0, int x_stride1, int w_stride0, int w_stride1) {
        /*
        X is n,m 
        W is m,p 
        indexes is 2,n_indexes
        z is n,p
        */
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Xbuf0[BLOCK_SIZE];
    int index_position = blockIdx.x * NTHREADS + threadIdx.x;
    if (index_position >= n_indexes) return;

    int tx = threadIdx.y;
    int posy = indexes[index_position]; // this is actually the minibatch index
    int posx = indexes[index_position + n_indexes] * BLOCK_SIZE + threadIdx.y;
    double acc0 = 0;
    for (int ks=0;ks<m;ks+=BLOCK_SIZE){
        Xbuf0[tx] = X[posy * x_stride0 + (tx + ks) * x_stride1];
        __syncthreads();
        for (int k=ks;k<ks+BLOCK_SIZE;++k){
            if (k >= m) { break;}
            double a = Xbuf0[k-ks];
            double b = W[(k * w_stride0) + (posx * w_stride1)];
            acc0 += a * b;
            //acc0 += Xbuf0[k-ks] * W[(k * w_stride0) + (posx * w_stride1)];
        }
        __syncthreads();
    }
    z[posy * p + posx] = acc0;
    
}""" % (self.block_size)


    def c_code(self, node, name, inp, out, sub):
        A,B,indexes,indexes_data,n,m,p,dropout = inp
        z,indexes_out,indexes_data_out = out
        fail = sub['fail']
        s = """

        float *X_data = CudaNdarray_DEV_DATA(%(A)s);
        float *W_data = CudaNdarray_DEV_DATA(%(B)s);
        float *I_data = CudaNdarray_DEV_DATA(%(indexes)s);
        int dims[] = {0,0,0};
        float *O_data; // output data

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        dims[2] = ((dtype_%(p)s*)PyArray_DATA(%(p)s))[0];
        int total_size = dims[0] * dims[2] * sizeof(float);
        int zdims[] = {0,0};
        zdims[0] = dims[0];
        zdims[1] = dims[2];

        %(indexes_out)s = %(indexes)s;
        Py_XINCREF(%(indexes_out)s);
        %(indexes_data_out)s = %(indexes_data)s;
        Py_XINCREF(%(indexes_data_out)s);

        const int* index_dims = CudaNdarray_HOST_DIMS(%(indexes)s);
        const int* X_strides = CudaNdarray_HOST_STRIDES(%(A)s);
        const int* W_strides = CudaNdarray_HOST_STRIDES(%(B)s);
        

        int grid_size = index_dims[1] / NTHREADS;
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless excatly a fit (modulo is zero)
        if (index_dims[1] %% (NTHREADS) != 0) grid_size++;

        cudaError_t sts;
        void * orig_z = %(z)s;
        if (CudaNdarray_prep_output(&%(z)s, 2, zdims))
        {
            %(fail)s;
        }
        //printf("0-fill %%d\\n",total_size);
        sts = cudaMemset(CudaNdarray_DEV_DATA(%(z)s), 0, total_size);
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_MemoryError,
                         "GpuEye: Error in memset %%d bytes of device memory.",
                         total_size);
            if(orig_z == NULL)
                Py_XDECREF(%(z)s);
            %(fail)s;
        }
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        O_data = CudaNdarray_DEV_DATA(%(z)s);

        if (grid_size > 0){
        sparsedot_fs<<<grid_size, blocks>>>(X_data, W_data, I_data, O_data,
                                         dims[0], dims[1], dims[2], index_dims[1],
                                         X_strides[0], X_strides[1], W_strides[0], W_strides[1]); 
        }
        CNDA_THREAD_SYNC;


        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: sparse_dot: %%s. n=%%d, m=%%d, p=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), wstrides=(%%d %%d)",
                    cudaGetErrorString(sts),
                    dims[0], dims[1], dims[2], grid_size, blocks.x, blocks.y,
                    index_dims[0], index_dims[1],
                    W_strides[0], W_strides[1]);
            %(fail)s;
        }
        """ % locals()
        return s

    #def c_code_cache_version(self):
    #    return (2,self.block_size)

    def __str__(self):
        return "GPUSparseDot_FS"

































class _GPUSparseDot_SparseByFull(theano.sandbox.cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, *args):
        if len(args) == 4:
            A,_,B,block_size = args
            aindexes,aindexes_data = _
        else:
            raise ValueError("Wrong number of arguments (expected 4)",args)
            
            
        self.block_size = block_size
        if A.dtype!='float32':
            A = T.cast(A, 'float32')
        if B.dtype!='float32':
            B = T.cast(B, 'float32')
        A,B = [safe_to_gpu(i) for i in [A,B]]
        n,m,p = [T.as_tensor_variable(i) for i in [A.shape[0],B.shape[0],B.shape[1]]]
        return theano.Apply(self,
                            inputs=[A,aindexes,aindexes_data, 
                                    B,
                                    n,m,p],
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,0))()])


    def grad(self, inputs, outputs):
        a,ia,iadata,b,n,m,p = inputs
        gz, = outputs
        
        agrad = T.dot(gz, b.T)
        bgrad = T.dot(a.T, gz)
        disc = theano.gradient.DisconnectedType()
        return tuple([agrad,disc(),disc(),bgrad] + 
                     [disc() for i in [n,m,p]])
    def connection_pattern(self, node):
        return [[True], [False], [False],
                [True],
                [False], [False], [False]]

    def c_support_code(self):
        return """

__global__ void sparsedot_sf(float* A, float* B, int* aindexes, int* aindexes_data,
        float* C, int n, int m, int p, 
        int n_aindexes,
        int a_stride0, int a_stride1, int b_stride0, int b_stride1) {
        /*
        A is n,m  is sparse
        B is m,p  has no dropout
        aindexes is 2,n_aindexes
        aindexes_data is n,2 = [...,[start_index, number_of_blocks],...]
        C is n,p
        */
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Abuf0[BLOCK_SIZE];
    int tx = threadIdx.y;
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    // this is the position of the computed value in C
    int posy = blockIdx.x;
    int posx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    if (posy >= n || posx >= p) return; 

    // this is where the list of positions in aindexes start (there are number_of_blocks positions)
    const int positions_start = aindexes_data[posy * 2] + n_aindexes;
    // this is the number of blocks that are non-zero in A
    const int number_of_blocks = aindexes_data[posy * 2 + 1];
    double acc0 = 0;
    for (int ks=0;ks<number_of_blocks;ks++){
        int aposx = aindexes[positions_start + ks] * BLOCK_SIZE;
        //Abuf0[tx] = A[posy * a_stride0 + (tx + aposx) * a_stride1];
        //printf("%%d %%d %%d %%d   tx %%d b %%d %%d aposx %%d A %%f\\n", posy, posx, positions_start - n_aindexes, number_of_blocks, tx, blockIdx.x, blockIdx.y, aposx, Abuf0[tx]);
        __syncthreads();
        for (int k=0;k<BLOCK_SIZE;++k){
            if (aposx+k >= m) { break;}
            // cant do this if p is not a multiple of block_size
            //double a = Abuf0[k];
            double a =  A[posy * a_stride0 + (k + aposx) * a_stride1];
            double b = B[((k+aposx) * b_stride0) + (posx * b_stride1)];
            acc0 += a * b;
        }
        __syncthreads();
    }
    C[posy * p + posx] = acc0;
    
}""" % (self.block_size)


    def c_code(self, node, name, inp, out, sub):
        A,aindexes,aindexes_data,B,n,m,p = inp
        z, = out
        fail = sub['fail']
        s = """
        float *A_data = CudaNdarray_DEV_DATA(%(A)s);
        float *B_data = CudaNdarray_DEV_DATA(%(B)s);
        int *AI_data  = (int*)CudaNdarray_DEV_DATA(%(aindexes)s);
        int *AID_data = (int*)CudaNdarray_DEV_DATA(%(aindexes_data)s);
        int dims[] = {0,0,0};
        float *O_data; // output data

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        dims[2] = ((dtype_%(p)s*)PyArray_DATA(%(p)s))[0];
        int total_size = dims[0] * dims[2] * sizeof(float);
        int zdims[] = {0,0};
        zdims[0] = dims[0];
        zdims[1] = dims[2];

        const int* aindex_dims = CudaNdarray_HOST_DIMS(%(aindexes)s);
        const int* A_strides = CudaNdarray_HOST_STRIDES(%(A)s);
        const int* B_strides = CudaNdarray_HOST_STRIDES(%(B)s);
        

        //int grid_size = dims[0] * dims[2] / NTHREADS / BLOCK_SIZE;
        dim3 grid_size(dims[0], dims[2] / BLOCK_SIZE);
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless exactly a fit (modulo is zero)
        if (dims[2] %% (BLOCK_SIZE) != 0) grid_size.y++;

        cudaError_t sts;
        void * orig_z = %(z)s;
        if (CudaNdarray_prep_output(&%(z)s, 2, zdims))
        {
            %(fail)s;
        }

        sts = cudaMemset(CudaNdarray_DEV_DATA(%(z)s), 0, total_size);
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_MemoryError,
                         "GpuEye: Error in memset %%d bytes of device memory.",
                         total_size);
            if(orig_z == NULL)
                Py_XDECREF(%(z)s);
            %(fail)s;
        }
        // todo: test if this makes a difference
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        O_data = CudaNdarray_DEV_DATA(%(z)s);

        if (grid_size.x * grid_size.y > 0){
        sparsedot_sf<<<grid_size, blocks>>>(A_data, B_data, AI_data, AID_data, O_data,
                                         dims[0], dims[1], dims[2], aindex_dims[1],
                                         A_strides[0], A_strides[1], B_strides[0], B_strides[1]); 
        }
        CNDA_THREAD_SYNC;

        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: sparse_dot_sf: %%s. n=%%d, m=%%d, p=%%d. grid=(%%d %%dx%%d), aindexes=(%%d %%d), bstrides=(%%d %%d)",
                    cudaGetErrorString(sts),
                    dims[0], dims[1], dims[2], grid_size, blocks.x, blocks.y,
                    aindex_dims[0], aindex_dims[1],
                    B_strides[0], B_strides[1]);
            %(fail)s;
        }
        """ % locals()
        return s

    #def c_code_cache_version(self):
    #    return (2,self.block_size)

    def __str__(self):
        return "GPUSparseDot_SF"










def gpu_sparse_dot_fs_equiv(A,B,d,s):
    ss = (A.shape[0], A.shape[1]/s, s)
    sd = (A.shape[0], A.shape[1]/s, 1)
    so = (A.shape[0], B.shape[1])
    return (T.dot(A,B).reshape(ss) * d.reshape(sd)).reshape(so)

def gpu_sparse_dot_ss_equiv(A,aD,B,bD,s):
    ss = (A.shape[0], A.shape[1]/s, s)
    sd = (A.shape[0], B.shape[1]/s, 1)
    sb = (A.shape[0], B.shape[1]/s, s)
    so = (A.shape[0], B.shape[1])
    cA = (A.reshape(ss) * aD.dimshuffle(0,1,'x')).reshape(A.shape)
    return (T.dot(cA,B).reshape(sb) * bD.reshape(sd)).reshape(so)


def mask_mult(a,m):
    return (a.reshape((a.shape[0], m.shape[1], -1)) * m.dimshuffle(0,1,'x')).reshape(a.shape)
    


if __name__ == '__main__':
    a = T.matrix('a')
    am = T.matrix('am')
    b = T.matrix('b')
    om = T.matrix('om')
    c = T.vector('c')

    blocksize = 16
    n = 32
    m = blocksize*32*4
    t = blocksize*32*4

    print (n,m,t),blocksize
    print "All these numbers should be 0 or close to 0:"


    z = _GPUSparseDot_SparseBySparse(blocksize)(a,am,b,om,c)
    p = mask_mult(T.dot(mask_mult(a,am), b) + c, om)

    f = theano.function([a,am,b,om,c], [z,p],allow_input_downcast=True)

    a = numpy.random.rand(n,m)
    am = numpy.random.rand(n,m/blocksize) < 0.25
    b = numpy.random.rand(m,t) 
    om = numpy.random.rand(n,t/blocksize) < 0.25
    c = numpy.random.rand(t)

    z,p = f(a,am,b,om,c)

    q = abs(z-p)
    print q.min(), q.max(), q.mean()
    


    # testing grads
    a = T.matrix('a')
    am = T.matrix('am')
    b = T.matrix('b')
    om = T.matrix('om')
    c = T.vector('c')


    z = _GPUSparseDot_SparseBySparse(blocksize)(a,am,b,om,c)
    p = mask_mult(T.dot(mask_mult(a,am), b) + c, om)
    

    zgrads = T.grad(z.sum(), [a,b,c])
    pgrads = T.grad(p.sum(), [a,b,c])

    f = theano.function([a,am,b,om,c], zgrads+pgrads, allow_input_downcast=True)

    a = numpy.random.rand(n,m)
    am = numpy.random.rand(n,m/blocksize) < 0.5
    b = numpy.random.rand(m,t) 
    om = numpy.random.rand(n,t/blocksize) < 0.5
    c = numpy.random.rand(t)

    grads = f(a,am,b,om,c)

    for i in range(3):
        q = abs(grads[i]-grads[i+3])
        print q.min(),q.max(), q.mean()

