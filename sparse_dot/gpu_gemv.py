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










class SSGemv_Rect(theano.sandbox.cuda.GpuOp):
    def __init__(self, block_size, inplace=False, do_rect=True):
        self.block_size = block_size
        self.do_rect = do_rect
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [5]}
        
    def __eq__(self, other):
        return type(self) == type(other) and self.block_size == other.block_size and self.do_rect == other.do_rect and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self))+hash(self.block_size)+hash(self.do_rect)+hash(self.inplace)

    def make_node(self, a, amask, B, bmask, c, output=None):
        """ 
        returns o = rect(aB+c) given the masks 
        so it's equivalent to computing
           a_masked = (a.reshape((-1,block_size)) * amask.dimshuffle(0,'x')).flatten()
           dot_result = maxmimum(dot(amasked, B) + c, 0)
           output = (dot_result.reshape((-1,block_size)) * bmask.dimshuffle(0,'x')).flatten()
        """
        if output:
            self.o = [output]
        else:
            self.o = []
        
        assert a.ndim == 1
        assert B.ndim == 2

        if a.dtype!='float32':
            A = T.cast(A, 'float32')
        if B.dtype!='float32':
            B = T.cast(B, 'float32')
        a,B,amask,bmask,c = [safe_to_gpu(i) for i in [a,B,amask,bmask,c]]
        return theano.Apply(self,
                            inputs=[a,amask,B,bmask,c]+self.o,
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,))()])

    def __str__(self):
        return "GpuSS_Gemv_Rect[%d,%d%s]"%(self.block_size,self.do_rect,",addinplace" if self.inplace else "")
    def c_code_cache_version(self):
        return (3,self.block_size)
    

    def grad(self, inputs, outputs):
        o = self(*(inputs))
        a,am,B,bm,c = inputs
        go, = outputs
        go_rectp = go * T.cast((o > 0), 'float32')
        # 
        ga = SSGemv_Rect(self.block_size,do_rect=False)(go_rectp, bm, B.T, am, T.zeros(1))
        #  equivalent to:
        #ga = T.dot(go_rectp, B.T).reshape((-1,self.block_size)) * am.dimshuffle(0,'x')
        #ga = ga.reshape(a.shape)

        # this could be optimized by multing only where the mask is nonzero
        a_masked = (a.reshape((-1,self.block_size)) * am.dimshuffle(0,'x')).flatten()
        gb = go_rectp.dimshuffle('x',0) * a_masked.dimshuffle(0,'x')
        # but doing so... doesn't take advantage of the inplace theano gimmicks, so its not worth.
        # The FSouter is faster, but idk how to use Composite it with stuff
        gb = _FSOuter()(go_rectp, a, am, self.block_size)

        # the easy one!
        gc = go_rectp

        disc = theano.gradient.DisconnectedType()
        return [ga, disc(), gb, disc(), gc]

    def connection_pattern(self, node): 
        # a, aidx, 
        # b, bidx, 
        # dropout
        return [[True], [False], 
                [True], [False],
                [True]]


    def c_support_code(self):
        op = "+=" if self.inplace else "="
        if self.do_rect:
            act = "O[posx] %s fmax(acc0 + C[posx], 0);"%op
        else:
            act = "O[posx] %s acc0;"%op #  + C[posx]; should be there but... not needed for its current use, should add doc about this!
            
        return """

__global__ void sparsedot_ss(
        float* A, float* B, int* aindexes, int* bindexes, 
        float* C,
        float* O, 
        int n, int m,
        int n_aindexes, int n_bindexes,
        int b_stride0, int b_stride1) {
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Abuf0[BLOCK_SIZE];
    int index_position = blockIdx.x * NTHREADS + threadIdx.x;
    if (index_position >= n_bindexes) return;
    int tx = threadIdx.y;

    // this is the position of the computed value in O
    int posx = bindexes[index_position] * BLOCK_SIZE + threadIdx.y;

    double acc0 = 0;
    for (int ks=0;ks<n_aindexes;ks++){
        int aposx = aindexes[ks] * BLOCK_SIZE;
        Abuf0[tx] = A[tx + aposx];
        __syncthreads();
        #pragma unroll BLOCK_SIZE
        for (int k=0;k<BLOCK_SIZE;++k){
            double a = Abuf0[k];
            double b = B[((k+aposx) * b_stride0) + (posx * b_stride1)];
            acc0 += a * b;
        }
        __syncthreads();
    }
    %s
    
}""" % (self.block_size,act)


    def c_code(self, node, name, inp, out, sub):
        if self.inplace:
            A,amask,B,bmask,C,O_inplace = inp
            O, = out
        else:
            A,amask,B,bmask,C = inp
            O, = out
            O_inplace = ""
        fail = sub['fail']
        inplace_def = "#define IS_INPLACE" if self.inplace else ""
        s = """
{
%(inplace_def)s
        cudaError_t sts;
        float *A_data = CudaNdarray_DEV_DATA(%(A)s);
        float *B_data = CudaNdarray_DEV_DATA(%(B)s);
        float *C_data = CudaNdarray_DEV_DATA(%(C)s);
        float *Amask_data  = (float*)CudaNdarray_DEV_DATA(%(amask)s);
        float *Bmask_data  = (float*)CudaNdarray_DEV_DATA(%(bmask)s);
        const int* A_dims = CudaNdarray_HOST_DIMS(%(A)s);
        const int* B_dims = CudaNdarray_HOST_DIMS(%(B)s);
        const int* C_dims = CudaNdarray_HOST_DIMS(%(C)s);
        const int* Amask_dims = CudaNdarray_HOST_DIMS(%(amask)s);
        const int* Bmask_dims = CudaNdarray_HOST_DIMS(%(bmask)s);

        const int* B_strides = CudaNdarray_HOST_STRIDES(%(B)s);

        const int O_dims[] = {B_dims[1]};
        int size_of_O_in_bytes = B_dims[1] * sizeof(float);
        float *O_data; // output data
        int *Aindexes_data; 
	int *Bindexes_data; 


        // A mask into indexes
        float * cpu_amask = (float*)malloc(sizeof(float) * Amask_dims[0]); // freed
        cudaMemcpy(cpu_amask, Amask_data, sizeof(float) * Amask_dims[0], cudaMemcpyDeviceToHost);

        int n_Anonzeros = 0, A_size = Amask_dims[0];
        int *A_nonzero_positions = (int*)malloc(A_size * sizeof(int)); // freed
        int *p = A_nonzero_positions;

        float *Aptr = cpu_amask;
        for (int i=0; i<A_size; i++){ if (*Aptr++ != 0.0){ *p++ = i; n_Anonzeros++;}}
        cudaMalloc(&Aindexes_data, sizeof(int) * n_Anonzeros); // freed
        cudaMemcpy(Aindexes_data, A_nonzero_positions, sizeof(int) * n_Anonzeros, cudaMemcpyHostToDevice);
        free(cpu_amask);
        free(A_nonzero_positions);


        // B mask into indexes
        float * cpu_bmask = (float*)malloc(sizeof(float) * Bmask_dims[0]); // freed
        cudaMemcpy(cpu_bmask, Bmask_data, sizeof(float) * Bmask_dims[0], cudaMemcpyDeviceToHost); 

        int n_Bnonzeros = 0, B_size = Bmask_dims[0];
        int *B_nonzero_positions = (int*)malloc(B_size * sizeof(int)); // freed
        p = B_nonzero_positions;

        float *Bptr = cpu_bmask;
        for (int i=0; i<B_size; i++){ if (*Bptr++ != 0.0){ *p++ = i; n_Bnonzeros++;}}
        cudaMalloc(&Bindexes_data, sizeof(int) * n_Bnonzeros); // freed
        cudaMemcpy(Bindexes_data, B_nonzero_positions, sizeof(int) * n_Bnonzeros, cudaMemcpyHostToDevice);
        free(cpu_bmask);
        free(B_nonzero_positions);


        // compute the grid shape
        int grid_size = n_Bnonzeros / NTHREADS;
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless excatly a fit (modulo is zero)
        if (n_Bnonzeros %% (NTHREADS) != 0) grid_size++;


        void * orig_O = %(O)s;
#ifdef IS_INPLACE
        const int* Oinplace_dims = CudaNdarray_HOST_DIMS(%(O_inplace)s);
        if (Oinplace_dims[0] != O_dims[0]){
        PyErr_Format(PyExc_RuntimeError,
                    "error: gemv_rect_inplace: dimensions for inplace add differ. %%dx1 (result) %%dx1 (inplace)\\n",
                    O_dims[0], Oinplace_dims[0]);
            %(fail)s;
        }

        O_data = CudaNdarray_DEV_DATA(%(O_inplace)s);
        %(O)s = %(O_inplace)s;
        Py_XINCREF(%(O)s);
        Py_XDECREF(orig_O);
#else
        if (CudaNdarray_prep_output(&%(O)s, 1, O_dims)){  %(fail)s;  }

        O_data = CudaNdarray_DEV_DATA(%(O)s);

        sts = cudaMemset(O_data, 0, size_of_O_in_bytes);
        if (cudaSuccess != sts){
            if(orig_O == NULL) Py_XDECREF(%(O)s);
            %(fail)s;
        }
#endif
        
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  
	if (grid_size > 0){
	  sparsedot_ss<<<grid_size, blocks>>>(A_data, B_data, 
					      Aindexes_data, Bindexes_data,
					      C_data, O_data,
					      A_dims[0], B_dims[1], n_Anonzeros, n_Bnonzeros,
					      B_strides[0], B_strides[1]); 
	}
        CNDA_THREAD_SYNC;

        cudaFree(Aindexes_data);
        cudaFree(Bindexes_data);

        //printf("Elapsed time : %%f ms %%d %%d %%d %%d (%%d)\\n" ,elapsedTime, grid_size, blocks.x, blocks.y, blocks.z, BLOCK_SIZE);
  /*printf("sparse_dot: %%s. n=%%d, m=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), bstrides=(%%d %%d)\\n",
                    cudaGetErrorString(sts),
                    A_dims[0], B_dims[1], grid_size, blocks.x, blocks.y,
                    index_dims[0], index_dims[1],
                    B_strides[0], B_strides[1]);*/
        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: sparse_dot: %%s. n=%%d, m=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), bstrides=(%%d %%d)\\n",
                    cudaGetErrorString(sts),
                    A_dims[0], B_dims[1], grid_size, blocks.x, blocks.y,
                    n_Anonzeros, n_Bnonzeros,
                    B_strides[0], B_strides[1]);
            %(fail)s;
        }
}
        """ % locals()
        return s



class _FSOuter(theano.sandbox.cuda.GpuOp):
    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [3]}

    def __eq__(self, other):
        return type(self) == type(other) and self.block_size == other.block_size

    def __hash__(self):
        return hash(type(self))+hash(self.block_size)

    def make_node(self, *args):
        """
        This is equivalent to 
        (T.outer(a,b).reshape((-1,block_size)) * bmask.dimshuffle(0,'x')).flatten()
        """
        if self.inplace:
            a,b,bmask,o,block_size = args
            self.o = [o]
        else:
            a,b,bmask,block_size = args
            self.o = []

        assert a.ndim == 1
        assert b.ndim == 1

        self.block_size = block_size

        if a.dtype!='float32':
            a = T.cast(a, 'float32')
        if b.dtype!='float32':
            b = T.cast(b, 'float32')
        return theano.Apply(self,
                            inputs=[safe_to_gpu(i) for i in [a,b,bmask]+self.o],
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,0))()])

    def __str__(self):
        return "GpuFS_Outer[%d%s]"%(self.block_size,",inplace" if self.inplace else "")

    def c_code_cache_version(self):
        return (3,self.block_size)

    def c_support_code(self):
        return """

__global__ void outer_fs(
        float* A, float* B, int* bindexes, 
        float* O, 
        int n, int m, int n_bindexes) {
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Abuf0[BLOCK_SIZE];
    int index_position = blockIdx.x;
    // not needed normally: if (index_position >= n_bindexes) return;

    // this is the position of the computed value in O
    int posx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int posy = bindexes[index_position] * BLOCK_SIZE;

    for (int k=0;k<BLOCK_SIZE;k++){
        O[n * (posy + k) + posx] %s A[posx] * B[posy + k];
    }
    
}""" % (self.block_size, "+=" if self.inplace else "=")


    def c_code(self, node, name, inp, out, sub):
        if self.inplace:
            A,B,bmask,O_inplace = inp
            O, = out
        else:
            A,B,bmask, = inp
            O, = out
            O_inplace = ""
        fail = sub['fail']
        inplace_def = "#define IS_INPLACE" if self.inplace else ""
        try:
            s = """
%(inplace_def)s
        float *A_data = CudaNdarray_DEV_DATA(%(A)s);
        float *B_data = CudaNdarray_DEV_DATA(%(B)s);
        float *Bmask_data  = (float*)CudaNdarray_DEV_DATA(%(bmask)s);
        const int* A_dims = CudaNdarray_HOST_DIMS(%(A)s);
        const int* B_dims = CudaNdarray_HOST_DIMS(%(B)s);
        const int* Bmask_dims = CudaNdarray_HOST_DIMS(%(bmask)s);

        const int O_dims[] = {B_dims[0],A_dims[0]};
        float *O_data; // output data
        int size_of_O_in_bytes = B_dims[0] * A_dims[0] * sizeof(float);


        // B mask into indexes
        float * cpu_bmask = (float*)malloc(sizeof(float) * Bmask_dims[0]);  // freed
        cudaMemcpy(cpu_bmask, Bmask_data, sizeof(float) * Bmask_dims[0], cudaMemcpyDeviceToHost);

        int n_Bnonzeros = 0, B_size = Bmask_dims[0];
        int *B_nonzero_positions = (int*)malloc(B_size * sizeof(int)); // free
        int *p = B_nonzero_positions;

        float *Bptr = cpu_bmask;
        for (int i=0; i<B_size; i++){ if (*Bptr++ != 0.0){ *p++ = i; n_Bnonzeros++;}}
        int *Bindexes_data; cudaMalloc(&Bindexes_data, sizeof(int) * n_Bnonzeros); // freed
        cudaMemcpy(Bindexes_data, B_nonzero_positions, sizeof(int) * n_Bnonzeros, cudaMemcpyHostToDevice);
        free(cpu_bmask);
        free(B_nonzero_positions);


        // compute the grid shape
        dim3 grid(n_Bnonzeros, A_dims[0] / BLOCK_SIZE / NTHREADS);
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless excatly a fit (modulo is zero)
        // FIX? this shouldnt be needed 
        // if (n_Bnonzeros %% (NTHREADS) != 0) grid_size++;

        cudaError_t sts;
        void * orig_O = %(O)s;

#ifdef IS_INPLACE
        const int* Oinplace_dims = CudaNdarray_HOST_DIMS(%(O_inplace)s);
        if (Oinplace_dims[0] != B_dims[0] ||
            Oinplace_dims[1] != A_dims[0]){
        PyErr_Format(PyExc_RuntimeError,
                    "error: outer_fs_inplace: dimensions for inplace add differ. %%dx%%d (result) %%dx%%d (inplace)\\n",
                    B_dims[0], A_dims[0], Oinplace_dims[0], Oinplace_dims[1]);
            %(fail)s;
        }

        O_data = CudaNdarray_DEV_DATA(%(O_inplace)s);
        %(O)s = %(O_inplace)s;
        Py_XINCREF(%(O)s);
        Py_XDECREF(orig_O);
#else
        if (CudaNdarray_prep_output(&%(O)s, 2, O_dims)){  %(fail)s;  }

        O_data = CudaNdarray_DEV_DATA(%(O)s);
        sts = cudaMemset(O_data, 0, size_of_O_in_bytes);
        if (cudaSuccess != sts){
            if(orig_O == NULL) Py_XDECREF(%(O)s);
            %(fail)s;
        }
#endif

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        if (grid.x > 0){
            outer_fs<<<grid, blocks>>>(A_data, B_data, 
                                            Bindexes_data,
                                            O_data,
                                            B_dims[0], A_dims[0], n_Bnonzeros); 
        }
        CNDA_THREAD_SYNC;

        cudaFree(Bindexes_data);
        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: outer_fs: %%s. n=%%d, m=%%d. grid=(%%dx%%d %%dx%%d), indexes=(%%d)\\n",
                    cudaGetErrorString(sts),
                    A_dims[0], B_dims[0], grid.x, grid.y, blocks.x, blocks.y,
                    n_Bnonzeros);
            %(fail)s;
        }
        """ % locals()
        except Exception,e:
            print e
            import traceback
            traceback.print_exc()
        return s



@theano.sandbox.cuda.opt.register_opt()
@theano.sandbox.cuda.opt.local_optimizer([theano.sandbox.cuda.GpuElemwise])
def local_inplace_fs_outer(node):
    if not ("Outer" in str(node) and isinstance(node.op, theano.sandbox.cuda.GpuElemwise)):
        return
    if node.op.scalar_op == theano.scalar.add:
        if node.inputs[0].owner and isinstance(node.inputs[0].owner.op, _FSOuter):
            print "Found",node
            own = node.inputs[0].owner
            if own.op.inplace: return
            inputs = own.inputs+[node.inputs[1],own.op.block_size]
            return [_FSOuter(True)(*inputs)]
        if node.inputs[1].owner and isinstance(node.inputs[1].owner.op, _FSOuter):
            print "Found",node
            raise "ImplementMe!"

@theano.sandbox.cuda.opt.register_opt()
@theano.sandbox.cuda.opt.local_optimizer(None)
def testopt(node):
    if not ("SS_Gemv" in str(node)):
        return
    if isinstance(node.op, theano.sandbox.cuda.GpuElemwise) and \
       node.op.scalar_op == theano.scalar.add:
        if node.inputs[0].owner and isinstance(node.inputs[0].owner.op, SSGemv_Rect):
            print "Found",node
            own = node.inputs[0].owner
            if own.op.inplace: return
            inputs = own.inputs+[node.inputs[1]]
            return [SSGemv_Rect(own.op.block_size,inplace=True,do_rect=own.op.do_rect)(*inputs)]
        if node.inputs[1].owner and isinstance(node.inputs[1].owner.op, SSGemv_Rect):
            print "Found",node
            own = node.inputs[1].owner
            if own.op.inplace: return
            inputs = own.inputs+[node.inputs[0]]
            return [SSGemv_Rect(own.op.block_size,inplace=True,do_rect=own.op.do_rect)(*inputs)]
    elif not isinstance(node.op, SSGemv_Rect):
        pass

if __name__ == "__main__" and 0:
    a = T.vector()
    b = T.vector()
    c = T.matrix()
    
    d = c * 2
    
    e = _FSOuter()(a,b,T.ones_like(b),1) + d
    e_ = T.outer(a,b) + d
    f = theano.function([a,b,c], [e,e_],allow_input_downcast=True)
    r = f([1,2,3],[1,2,3],numpy.eye(3))
    print r[0]
    print r[1]


if __name__ == "__main__" and 0:
    a = T.vector()
    b = T.vector()
    c = T.matrix()
    
    d = c * 2
    
    e = _FSOuter()(a,b,T.ones([b.shape[0]/64]),64) + d
    f = theano.function([a,b,c], [e],allow_input_downcast=True)
    for i in range(1000):
        for j in range(10000):
            r = f(numpy.random.random((64*10)),numpy.random.random((64*10)),
                  numpy.random.random((64*10,64*10)))
        raw_input("step")


if __name__ == "__main__" and 0:
    bs = 64
    rate = 0.250
    n = 4096
    m = 4096
    print n / bs * rate

    a = T.vector('a')
    am = T.vector('am')
    bm = T.vector('bm')
    c = T.vector('c')
    b = theano.shared(numpy.float32(numpy.random.random((n, m)) - 0.5),'bshared')

    o = SSGemv_Rect(bs)(a,am,b,bm,c)

    f = theano.function([a,am,bm,c],
                        [o+0], 
                        profile=0, allow_input_downcast=True)

    rbin = lambda n: numpy.random.uniform(0,1,(n,)) < rate
    for j in range(1000):
        for i in range(50000):
            f(numpy.random.random(n) - 0.5, rbin(n / bs), rbin(m / bs),
              numpy.random.random(m) - 0.5)
        raw_input("step")


if __name__ == "__main__" and 0:
    
    a = T.vector('a')
    b = T.matrix('b')
    ad_ = T.vector('ad')
    ad = DualVectorNonzero()(ad_)
    bd = T.vector('bd')

    c,_ = _GPUSparseGemv_SparseBySparse()(a,ad,b,bd,2)

    f = theano.function([a,b,ad_,bd],[c+0, T.dot(a,b)])
    
    c,d = f([1,2], [[1,2,3,4],[4,5,6,7]], [1],[1,1])

    print c
    print d


if __name__ == "__main__":
    
    if 0:
        a = T.vector('a')
        am = T.vector('am')
        b = T.matrix('b')
        bm = T.vector('bm')
        c = T.vector('c')
        o = SSGemv_Rect(2)(a,am,b,bm,c)
        asp = (a.reshape((-1,2)) * am.dimshuffle(0,'x')).flatten()
        ko = (T.maximum(T.dot(asp,b)+c,0).reshape((-1,2)) * bm.dimshuffle(0,'x')).flatten()

        f = theano.function([a,am,b,bm,c],[o+0,ko])
        x,y= f([1,-2,-3,4],[1,0],
               [[1,2],[3,-4],[5,6],[7,-8]],[1],
               [1,4])
        print x
        print y

    bs = 64
    rate = 0.250
    n = 4096
    m = 4096
    print n / bs * rate

    a = T.vector('a')
    am = T.vector('am')
    bm = T.vector('bm')
    c = T.vector('c')
    b = theano.shared(numpy.float32(numpy.random.random((n, m)) - 0.5),'bshared')

    o = SSGemv_Rect(bs)(a,am,b,bm,c)
    asp = (a.reshape((-1,bs)) * am.dimshuffle(0,'x')).flatten()
    ko = (T.maximum(T.dot(asp,b)+c,0).reshape((-1,bs)) * bm.dimshuffle(0,'x')).flatten()

    f = theano.function([a,am,bm,c],
                        [o+0,ko+0], 
                        profile=0, allow_input_downcast=True)

    rbin = lambda n: numpy.random.uniform(0,1,(n,)) < rate
    import time
    t0 = time.time()
    for i in range(1000):
        x,y = f(numpy.random.random(n) - 0.5, rbin(n / bs), rbin(m / bs),
                numpy.random.random(m) - 0.5)
    t1 = time.time()
    print abs(x-y).sum(),abs(x).sum(),abs(y).sum()
    print t1-t0


    grads = T.grad(o.sum(), [a,c,b])
    kgrads = T.grad(ko.sum(), [a,c,b])
    updates = [(b,b-0.1*grads[2])]

    f = theano.function([a,am,bm,c],
                        [i+0 for i in grads+kgrads],
                        allow_input_downcast=True,
                        updates = updates,
                        profile=0)
    
    t0 = time.time()
    for i in range(10):
        gs = f(numpy.random.random(n) - 0.5, rbin(n / bs), rbin(m / bs),
               numpy.random.random(m) - 0.5)
    for i,j in zip(gs[:3],gs[3:]):
        print abs(i-j).sum(),abs(i).sum(),abs(j).sum()
    t1 = time.time()
    print t1-t0
