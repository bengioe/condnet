import theano
import theano.tensor as T




class Gemm_ss(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other) and self.blocksize == other.blocksize

    def __hash__(self):
        return hash(type(self)) + hash(self.blocksize)

    def __init__(self, blocksize):
        self.blocksize = blocksize

        
    def make_node(self, a, amask, b, omask, c):
        """ compute O = AB + c given the masks on A and O """
        amask = T.cast(amask, 'int8')
        omask = T.cast(omask, 'int8')
        return theano.Apply(self,
                            inputs=[a,amask,b,omask,c],
                            outputs=[a.type()])


    def grad(self, inputs, outputs):
        a,am,b,om,c = inputs
        gz, = outputs
        
        gzo = (gz.reshape((gz.shape[0], om.shape[1], -1)) * om.dimshuffle(0,1,'x')).reshape(gz.shape)
        # this can be rewritten as a sparse_dot:
        #  xgrad = T.dot(gzo, b.T)
        #  xgrad = (xgrad.reshape((am.shape[0], am.shape[1], -1)) * am.dimshuffle(0,1,'x')).reshape(xgrad.shape)
        #xgrad = sparse_dot(gz, om, b.T, am, T.zeros_like(c), self.blocksize)
        xgrad = Gemm_ss(self.blocksize)(gz,om,b.T,am,T.as_tensor_variable(0.))
        
        #  this is inefficient:
        #ygrad = sparse_dot(a.T, am.T, gzo, None, T.zeros_like(a), self.blocksize)
        #  because A is "row-sparse" meaning that it has rows with
        #  blocks of zeros and non-zeros, while columns aren't sparse
        #  in blocks, so when using the transpose it would be
        #  pointless because we would have to check every row entry
        #  and thus it is equivalent to doing a full gemm, so let
        #  theano handle this one.
        a_masked = (a.reshape((am.shape[0], am.shape[1], -1)) * am.dimshuffle(0,1,'x')).reshape(a.shape)
        ygrad = T.dot(a_masked.T, gzo)
        
        ograd = T.cast(T.zeros_like(om),'float32')
        dt = theano.gradient.DisconnectedType()
        cgrad = gzo.sum(axis=0)
        return tuple([xgrad,dt(),ygrad,ograd,cgrad])

    def connection_pattern(self, node):
        return map(lambda x:[x],[True,False,True,True,True])

    #def c_code_cache_version(self):
    #    return (1,self.block_size)

    def __str__(self):
        return "SparseGemm_ss"


    def c_code(self, node, name, inp, out, sub):
        a,amask,b,omask,c = inp
        o, = out
        fail = sub['fail']
        blocksize = self.blocksize
        s = """
        
        float* A_data = (float*)PyArray_DATA(%(a)s);
        int8_t* amask = (int8_t*)PyArray_DATA(%(amask)s);
        float* B_data = (float*)PyArray_DATA(%(b)s);
        int8_t* omask = (int8_t*)PyArray_DATA(%(omask)s);
        float* C_data = (float*)PyArray_DATA(%(c)s);
        npy_intp* _A_strides = PyArray_STRIDES(%(a)s);
        npy_intp* _B_strides = PyArray_STRIDES(%(b)s);
        npy_intp* _amask_strides = PyArray_STRIDES(%(amask)s);
        npy_intp* _omask_strides = PyArray_STRIDES(%(omask)s);
        npy_intp* _c_strides = PyArray_STRIDES(%(c)s);
        npy_intp A_strides[2] = {_A_strides[0] / sizeof(float), _A_strides[1] / sizeof(float)};
        npy_intp B_strides[2] = {_B_strides[0] / sizeof(float), _B_strides[1] / sizeof(float)};
        npy_intp amask_strides[2] = {_amask_strides[0] / sizeof(int8_t), 
                                     _amask_strides[1] / sizeof(int8_t)};
        npy_intp omask_strides[2] = {_omask_strides[0] / sizeof(int8_t), 
                                     _omask_strides[1] / sizeof(int8_t)};
        npy_intp c_stride;
        if (PyArray_NDIM(%(c)s) == 0){
           c_stride = 0;
        } else{
           c_stride = _c_strides[0] / sizeof(float);
        }


        npy_intp odims[] = {PyArray_DIMS(%(a)s)[0], PyArray_DIMS(%(b)s)[1]};
        npy_intp cross_dim = PyArray_DIMS(%(a)s)[1];


        //printf("strides %%d %%d, %%d %%d, %%d %%d\\n shapes %%d %%d %%d %%d \\n", A_strides[0], A_strides[1], B_strides[0], B_strides[1], amask_strides[0], amask_strides[1], odims[0], cross_dim, PyArray_DIMS(%(b)s)[0], odims[1]);

        if (%(o)s == NULL){
          %(o)s = (PyArrayObject*)PyArray_ZEROS(2, odims, PyArray_TYPE(%(a)s), 0);
        }
        float* O_data = (float*)PyArray_DATA(%(o)s);
        
        int N = odims[0], M = odims[1], P = cross_dim;
        int bs = %(blocksize)s;
        
        //printf("%%d %%d %%d\\n",N,M,P);


        for (int y=0;y<N;y++){
          for (int x=0;x<M/bs;x++){
            if (omask[y*omask_strides[0] + x*omask_strides[1]] == 1){
              // compute the sparse vector dot product of "A[y,:].B[:,x]" block
              //int8_t* amp = amask + y * P / bs;
              
              // maybe it would be wiser to use doubles for accs
              double accs[bs];
              for(int i=0;i<bs;i++){accs[i] = C_data[(x*bs+i)*c_stride];}

              for (int p=0;p<P/bs;p++){
        //printf("%%d %%d %%d %%d\\n",y,x,p,*amp);
                if (amask[y*amask_strides[0]+p*amask_strides[1]] == 1){
                  for (int i=p*bs;i<p*bs+bs;i++){
                    float a_val = A_data[y * A_strides[0] + i * A_strides[1]];
                    float* b_val = &B_data[i * B_strides[0] + (x*bs) * B_strides[1]];
                    int strd = B_strides[1];
                    for (int k=0;k<bs;k++){
                      //if (y==0 && k==0){printf("%%d %%f\\n",k,B_data[i * B_strides[0] + (x*bs + k) * B_strides[1]]);}
                      accs[k] += a_val * (*b_val);
                      b_val += strd;
                }}}
              }
              
              for(int i=0;i<bs;i++){O_data[y*M+x*%(blocksize)s+i] = accs[i];}
            }
          }
        }

        


        """ % locals()
        return s



















class Gemm_fs(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other) and self.blocksize == other.blocksize

    def __hash__(self):
        return hash(type(self)) + hash(self.blocksize)

    def __init__(self, blocksize):
        self.blocksize = blocksize

        
    def make_node(self, a, b, omask, c):
        """ compute O = AB + c given the masks on O """
        omask = T.cast(omask, 'int8')

        return theano.Apply(self,
                            inputs=[a,b,omask,c],
                            outputs=[a.type()])


    def grad(self, inputs, outputs):
        a,b,om,c = inputs
        gz, = outputs
        
        gzo = (gz.reshape((gz.shape[0], om.shape[1], -1)) * om.dimshuffle(0,1,'x')).reshape(gz.shape)
        xgrad = T.dot(gzo, b.T)
        ygrad = T.dot(a.T, gzo)
        ograd = T.cast(T.zeros_like(om),'float32')
        dt = theano.gradient.DisconnectedType()
        cgrad = gzo.sum(axis=0)
        return tuple([xgrad,ygrad,ograd,cgrad])

    def connection_pattern(self, node):
        return map(lambda x:[x],[True,True,True,True])

    #def c_code_cache_version(self):
    #    return (1,self.block_size)

    def __str__(self):
        return "SparseGemm_fs"

    def c_support_code(self):
        return """

#include <ctime>
        """


    def c_code(self, node, name, inp, out, sub):
        a,b,omask,c = inp
        o, = out
        fail = sub['fail']
        blocksize = self.blocksize
        s = """
        //clock_t t0 = clock();
        float* A_data = (float*)PyArray_DATA(%(a)s);
        float* B_data = (float*)PyArray_DATA(%(b)s);
        int8_t* omask = (int8_t*)PyArray_DATA(%(omask)s);
        float* C_data = (float*)PyArray_DATA(%(c)s);
        npy_intp* _A_strides = PyArray_STRIDES(%(a)s);
        npy_intp* _B_strides = PyArray_STRIDES(%(b)s);
        npy_intp* _omask_strides = PyArray_STRIDES(%(omask)s);
        npy_intp* _c_strides = PyArray_STRIDES(%(c)s);
        npy_intp A_strides[2] = {_A_strides[0] / sizeof(float), _A_strides[1] / sizeof(float)};
        npy_intp B_strides[2] = {_B_strides[0] / sizeof(float), _B_strides[1] / sizeof(float)};
        npy_intp omask_strides[2] = {_omask_strides[0] / sizeof(int8_t), 
                                     _omask_strides[1] / sizeof(int8_t)};
        npy_intp c_stride;
        if (PyArray_NDIM(%(c)s) == 0){
           c_stride = 0;
        } else{
           c_stride = _c_strides[0] / sizeof(float);
        }
        //printf("strides %%d %%d, %%d %%d\\n", A_strides[0], A_strides[1], B_strides[0], B_strides[1]);
        npy_intp odims[] = {PyArray_DIMS(%(a)s)[0], PyArray_DIMS(%(b)s)[1]};
        npy_intp cross_dim = PyArray_DIMS(%(a)s)[1];


        if (%(o)s == NULL){
          %(o)s = (PyArrayObject*)PyArray_ZEROS(2, odims, PyArray_TYPE(%(a)s), 0);
        }
        float* O_data = (float*)PyArray_DATA(%(o)s);
        
        int N = odims[0], M = odims[1], P = cross_dim;
        int bs = %(blocksize)s;
        
        //printf("%%d %%d %%d %%d\\n",N,M,P,bs);
        //int asd = 0;
        for (int y=0;y<N;y++){
          for (int x=0;x<M/bs;x++){
            if (omask[y*omask_strides[0] + x*omask_strides[1]] == 1){
              // compute the vector dot product of "A[y,:].B[:,x]" block
              // maybe it would be wiser to use doubles for accs
              float accs[bs];
              for(int i=0;i<bs;i++){accs[i] = C_data[(x*bs+i)*c_stride];}
              //memcpy(accs, C_data + x * bs, bs * sizeof(float));

              for (int p=0;p<P/bs;p++){
                  for (int i=p*bs;i<p*bs+bs;i++){
                    float a_val = A_data[y * A_strides[0] + i * A_strides[1]];
                    float* b_val = &B_data[i * B_strides[0] + (x*bs) * B_strides[1]];
                    int strd = B_strides[1];
                    for (int k=0;k<bs;k++){
                      accs[k] += a_val * (*b_val);
                      b_val += strd;
        //asd++;
                }}
              }
              
              for(int i=0;i<bs;i++){O_data[y*M+x*%(blocksize)s+i] = accs[i];}
              //memcpy(O_data + y * M + x * bs, accs, bs * sizeof(float));
            }
          }
        }
        //clock_t t1 = clock();
        //printf("%%d %%f\\n",asd,double(t1-t0)/CLOCKS_PER_SEC);

        """ % locals()
        return s






class Gemm_sf(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other) and self.blocksize == other.blocksize

    def __hash__(self):
        return hash(type(self)) + hash(self.blocksize)

    def __init__(self, blocksize):
        self.blocksize = blocksize

        
    def make_node(self, a, amask, b, c):
        """ compute O = AB + c given the masks on A and O """
        amask = T.cast(amask, 'int8')
        return theano.Apply(self,
                            inputs=[a,amask,b,c],
                            outputs=[a.type()])


    def grad(self, inputs, outputs):
        a,am,b,c = inputs
        gz, = outputs
        
        gzo = gz#(gz.reshape((gz.shape[0], om.shape[1], -1)) * om.dimshuffle(0,1,'x')).reshape(gz.shape)
        xgrad = T.dot(gzo, b.T)
        xgrad = (xgrad.reshape((am.shape[0], am.shape[1], self.blocksize)) * am.dimshuffle(0,1,'x')).reshape(xgrad.shape)
        #xgrad = sparse_dot(gz, None, b.T, am, T.zeros_like(c), self.blocksize)
        #xgrad = Gemm_fs(self.blocksize)(gz, b.T, am, T.as_tensor_variable(0.))

        a_masked = (a.reshape((am.shape[0], am.shape[1], -1)) * am.dimshuffle(0,1,'x')).reshape(a.shape)
        ygrad = T.dot(a_masked.T, gzo)
        
        
        dt = theano.gradient.DisconnectedType()
        cgrad = gzo.sum(axis=0)
        return tuple([xgrad,dt(),ygrad,cgrad])

    def connection_pattern(self, node):
        return map(lambda x:[x],[True,False,True,True])

    #def c_code_cache_version(self):
    #    return (1,self.block_size)

    def __str__(self):
        return "SparseGemm_sf"


    def c_code(self, node, name, inp, out, sub):
        a,amask,b,c = inp
        o, = out
        fail = sub['fail']
        blocksize = self.blocksize
        s = """
        float* A_data = (float*)PyArray_DATA(%(a)s);
        int8_t* amask = (int8_t*)PyArray_DATA(%(amask)s);
        float* B_data = (float*)PyArray_DATA(%(b)s);
        float* C_data = (float*)PyArray_DATA(%(c)s);
        npy_intp* _A_strides = PyArray_STRIDES(%(a)s);
        npy_intp* _B_strides = PyArray_STRIDES(%(b)s);
        npy_intp* _amask_strides = PyArray_STRIDES(%(amask)s);
        npy_intp* _c_strides = PyArray_STRIDES(%(c)s);
        npy_intp A_strides[2] = {_A_strides[0] / sizeof(float), _A_strides[1] / sizeof(float)};
        npy_intp B_strides[2] = {_B_strides[0] / sizeof(float), _B_strides[1] / sizeof(float)};
        npy_intp amask_strides[2] = {_amask_strides[0] / sizeof(int8_t), 
                                     _amask_strides[1] / sizeof(int8_t)};
        npy_intp c_stride;
        if (PyArray_NDIM(%(c)s) == 0){
           c_stride = 0;
        } else{
           c_stride = _c_strides[0] / sizeof(float);
        }
        //printf("sf: strides %%d %%d, %%d %%d, %%d %%d\\n", A_strides[0], A_strides[1], B_strides[0], B_strides[1], amask_strides[0], amask_strides[1]);

        npy_intp odims[] = {PyArray_DIMS(%(a)s)[0], PyArray_DIMS(%(b)s)[1]};
        npy_intp cross_dim = PyArray_DIMS(%(a)s)[1];
        int output_dim = odims[1];
        //printf("sf: shapes  %%d %%d, %%d %%d\\n", odims[0], cross_dim, PyArray_DIMS(%(b)s)[0], odims[1]);


        if (%(o)s == NULL){
          %(o)s = (PyArrayObject*)PyArray_ZEROS(2, odims, PyArray_TYPE(%(a)s), 0);
        }
        float* O_data = (float*)PyArray_DATA(%(o)s);

        int N = odims[0], M = odims[1], P = cross_dim;
        int bs = %(blocksize)s;
        int niter = M/bs;
        if (M %% bs != 0){
          niter++;
        }
        

        //printf("%%d %%d %%d\\n",N,M,P);

        if (amask_strides[1] != 1 || A_strides[1] != 1){
           printf("Error: It seems you're doing a sparse-full dot product (A,B) with a transposed A, this is unimplemented because it is inefficient, as such the result of this Op will be wrong (because it is assumed that the row-sparse A is not transposed).\\n");
           %(fail)s;
        }

        for (int y=0;y<N;y++){
          for(int x=0;x<niter;x++){
              // compute the sparse vector dot product of "A[y,:].B[:,x]" block
              bs = std::min(%(blocksize)s,output_dim-x*%(blocksize)s);
              //printf("%%d %%d %%d %%d %%d\\n",bs,output_dim,x,niter,B_strides[1]);
              
              // maybe it would be wiser to use doubles for accs
              double accs[bs];
              for(int i=0;i<bs;i++){accs[i] = C_data[(x*%(blocksize)s+i)*c_stride];}
        
              for (int p=0;p<P/%(blocksize)s;p++){
                if (amask[y*amask_strides[0]+p*amask_strides[1]] == 1){
                  for (int i=p*%(blocksize)s;i<p*%(blocksize)s+%(blocksize)s;i++){
                    float a_val = A_data[y * A_strides[0] + i * A_strides[1]];
                    float* b_val = &B_data[i * B_strides[0] + (x*%(blocksize)s) * B_strides[1]];
                    int strd = B_strides[1];
                    for (int k=0;k<bs;k++){
                      accs[k] += a_val * (*b_val);
                      b_val += strd;
                }}}
              }
              for(int i=0;i<bs;i++){O_data[y*M+x*%(blocksize)s+i] = accs[i];}
              //memcpy(O_data + y * M + x * %(blocksize)s, accs, bs * sizeof(float));
            
          }
        }

        

        """ % locals()
        return s




def sparse_theano_ss(a, amask, b, omask, c, block_size):
    a = (a.reshape((a.shape[0],-1,block_size)) * amask.dimshuffle(0,1,'x')).reshape(a.shape)
    q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
    return q.reshape((a.shape[0], b.shape[1]))
def sparse_theano_fs(a, b, omask, c, block_size):
    q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
    return q.reshape((a.shape[0], b.shape[1]))
def sparse_theano_sf(a, amask, b, c, block_size):
    a = (a.reshape((a.shape[0],-1,block_size)) * amask.dimshuffle(0,1,'x')).reshape(a.shape)
    q = T.dot(a,b)+c
    return q

if __name__ == "__main__":
    bs = 32
    n,m,p = 64,1024,1024

    print (n,m,p),bs
    print "All these numbers should be 0, or close to 0:"

    import numpy
    A = T.matrix('a')
    B = T.matrix('b')
    am = T.matrix('am',dtype='int8')
    bm = T.matrix('bm',dtype='int8')
    c = T.vector('c')


    z = Gemm_ss(bs)(A,am,B,bm,c)
    zt= sparse_theano_ss(A,am,B,bm,c,bs)
    zfs = Gemm_fs(bs)(A,B,bm,c)
    zfst= sparse_theano_fs(A,B,bm,c,bs)
    zsf = Gemm_sf(bs)(A,am,B,c)
    zsft= sparse_theano_sf(A,am,B,c,bs)

    f = theano.function([A,B,am,bm,c],[z,zt,zfs,zfst,zsf,zsft],allow_input_downcast=True,profile=0)
    fsf = theano.function([A,B,am,c],[zsf,zsft],allow_input_downcast=True,profile=0)

    

    a = numpy.random.uniform(-1,1,(n,p))
    b = numpy.random.uniform(-1,1,(p,m))
    am = numpy.random.uniform(0,1,(n,p/bs)) < 0.105
    bm = numpy.random.uniform(0,1,(n,m/bs)) < 0.105
    c = numpy.random.uniform(-1,1,(m,))

    for i in range(10):
        z,zt,zfs,zfst,zsf,zsft = f(a,b,am,bm,c)
        if i == 0:
            print abs(z-zt).mean(),abs(zsf-zsft).mean(),abs(zfs-zfst).mean()
        if abs(z-zt).mean() > 0.0001 or abs(zfs-zfst).mean() > 0.0001 or abs(zsf-zsft).mean() > 0.0001:
            print abs(z-zt).sum(), abs(z-zt).mean()
            print abs(zfs-zfst).sum(), abs(zfs-zfst).mean()
            print abs(zsf-zsft).sum(), abs(zsf-zsft).mean()
            print z
            print zt
            raise Exception("Something seems off")
            
            
    # unaligned sparsefull
    M = m - bs/2

    b = numpy.random.uniform(-10,10,(p,M))
    c = numpy.random.uniform(-10,10,(M,))

    zsf,zsft = fsf(a,b,am,c)
    
    print abs(zsf-zsft).mean()
    if abs(zsf-zsft).mean() > 0.0001: 
        print b.shape
        print abs(zsf-zsft).sum(), abs(zsf-zsft).mean()
        print zsf
        print zsft
        raise Exception("Something seems off")
        

    


    # Grads

    A = T.matrix('a')
    B = T.matrix('b')
    am = T.matrix('am',dtype='int8')
    bm = T.matrix('bm',dtype='int8')
    c = T.vector('c')


    z = Gemm_ss(bs)(A,am,B,bm,c)
    zgrads = T.grad(T.tanh(z).sum(),[A,B,c])
    zt= sparse_theano_ss(A,am,B,bm,c,bs)
    tgrads = T.grad(T.tanh(zt).sum(),[A,B,c])

    zfs = Gemm_fs(bs)(A,B,bm,c)
    zfsgrads = T.grad(T.tanh(zfs).sum(),[A,B,c])
    ztfs= sparse_theano_fs(A,B,bm,c,bs)
    tfsgrads = T.grad(T.tanh(ztfs).sum(),[A,B,c])

    zsf = Gemm_sf(bs)(A,am,B,c)
    zsfgrads = T.grad(T.tanh(zsf).sum(),[A,B,c])
    ztsf= sparse_theano_sf(A,am,B,c,bs)
    tsfgrads = T.grad(T.tanh(ztsf).sum(),[A,B,c])

    f = theano.function([A,B,am,bm,c],zgrads+tgrads,allow_input_downcast=True)
    ffs = theano.function([A,B,bm,c],zfsgrads+tfsgrads,allow_input_downcast=True)
    fsf = theano.function([A,B,am,c],zsfgrads+tsfgrads,allow_input_downcast=True)

    
    a = numpy.random.uniform(-1,1,(n,p))
    b = numpy.random.uniform(-1,1,(p,m))
    am = numpy.random.uniform(0,1,(n,p/bs)) < 0.105
    bm = numpy.random.uniform(0,1,(n,m/bs)) < 0.105
    c = numpy.random.uniform(-1,1,(m,))

    grads = f(a,b,am,bm,c)
    
    for i in range(3):
        print abs(grads[i]-grads[i+3]).mean()
        if abs(grads[i]-grads[i+3]).mean() > 0.0001:
            raise Exception("Something seems off")

    grads = ffs(a,b,bm,c)

    for i in range(3):
        print abs(grads[i]-grads[i+3]).mean()
        if abs(grads[i]-grads[i+3]).mean() > 0.0001:
            raise Exception("Something seems off")

    grads = fsf(a,b,am,c)

    for i in range(3):
        print abs(grads[i]-grads[i+3]).mean()
        if abs(grads[i]-grads[i+3]).mean() > 0.0001:
            print i, abs(grads[i]-grads[i+3]).mean()
            raise Exception("Something seems off")

    # misaligned sparsefull

    bs = 32
    n,m,p = 64,1000,1024
    a = numpy.random.uniform(-1,1,(n,p))
    b = numpy.random.uniform(-1,1,(p,m))
    am = numpy.random.uniform(0,1,(n,p/bs)) < 0.105
    c = numpy.random.uniform(-1,1,(m,))


    grads = fsf(a,b,am,c)


    for i in range(3):
        print abs(grads[i]-grads[i+3]).mean()
        if abs(grads[i]-grads[i+3]).mean() > 0.0001:
            print i, abs(grads[i]-grads[i+3]).mean()
            raise Exception("Something seems off")

    
