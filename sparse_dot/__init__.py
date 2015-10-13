import theano
import theano.tensor as T

if 'cpu' in theano.config.device:
    sparse_trick = True
    from cpu_gemm import Gemm_ss, Gemm_sf, Gemm_fs

    def sparse_dot(a, amask, b, omask, c, block_size):
        if sparse_trick:
            if amask is not None and omask is not None:
                return Gemm_ss(block_size)(a,amask,b,omask,c)
            elif amask is not None and omask is None:
                return Gemm_sf(block_size)(a,amask,b,c)
            elif amask is None and omask is not None:
                #q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
                #return q.reshape((a.shape[0], b.shape[1]))
                return Gemm_fs(block_size)(a,b,omask,c)
            else:
                return T.dot(a,b)+c

        if omask is not None:
            q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
            return q.reshape((a.shape[0], b.shape[1]))
        else:
            return T.dot(a,b)+c

elif 'gpu' in theano.config.device:
    # so I updated theano and it broke the GPU implementation...
    # I don't have the time to go through it now but I will soon
    sparse_trick = False
    from gpu_gemm import _GPUSparseDot_SparseBySparse

    def sparse_dot(a, amask, b, omask, c, block_size):
        
        if sparse_trick:
            if amask is not None and omask is not None:
                return _GPUSparseDot_SparseBySparse(block_size)(a,amask,b,omask,c)
            elif amask is not None and omask is None:
                return T.dot(a,b)+c
            elif amask is None and omask is not None:
                q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
                return q.reshape((a.shape[0], b.shape[1]))
            else:
                return T.dot(a,b)+c

        if omask is not None:
            q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
            return q.reshape((a.shape[0], b.shape[1]))
        else:
            return T.dot(a,b)+c


def sparse_dot_theano(a, amask, b, omask, c, block_size):
    if omask is not None:
        q = (T.dot(a,b)+c).reshape((a.shape[0], -1, block_size)) * omask.dimshuffle(0,1,'x')
        return q.reshape((a.shape[0], b.shape[1]))
    else:
        return T.dot(a,b)+c
