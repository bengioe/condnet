
import numpy
import cPickle as pickle
import gzip
import uuid
import os
import time


from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import theano
import theano.tensor as T
theano.config.compute_test_value = 'off'


from sparse_dot import sparse_dot


def shared(shape, name, r=None):
    if r is None:
        r0 = r1 = numpy.sqrt(6. / numpy.sum(shape))
    elif r is "+":
        r0 = 0
        r1 = numpy.sqrt(6. / numpy.sum(shape))
    else:
        r0 = r1 = r
    values = numpy.float32(numpy.random.uniform(-r0,r1, shape))
    return theano.shared(values, name = name)


class SigmoidPolicyDropout:
    def __init__(self):
        self.hasGrad = True
    def __call__(self, input, input_mask, n_in, n_out, blocksize, rate, prew,n):
        self.n = n
        self.prew = prew
        self.n_in = n_in
        nblocks = n_out / blocksize

        prior_values = numpy.ones((nblocks,), dtype=theano.config.floatX) * rate
        prior = theano.shared(value=prior_values, name='q', borrow=True)
        theta = shared((n_in,n_out), 't')
        tb = shared((n_out), 'tb', 0)
        

        self.baseline = T.mean(T.nnet.sigmoid(T.dot(input,theta)+tb), axis=1)
        self.baseline_weights = [theta,tb]


        w = shared((n_in, nblocks), 'z')
        b = shared((nblocks,), 'c', 0)  
        self.weights = [w,b]

        #probs = T.nnet.sigmoid(T.dot(input, w) + b) * 0.99 + 0.01
        probs = T.nnet.sigmoid(sparse_dot(input, input_mask, w, None, b, blocksize)) * 0.99 + 0.01
        
        # necessary if using the log policy form:
        #probs = T.maximum(T.minimum(0.99,probs),0.0001)
        self.probs = probs

        rn = srng.uniform(size=(input.shape[0], nblocks), 
                                low=0.0, high=1.0,dtype='float32') 
        if 'gpu' in theano.config.device:
            mask = T.cast(rn < probs,'float32') 
        else:
            mask = rn < probs

        fmask = T.cast(mask, 'float32')
        # I'm not sure which of these forms is more numerically stable, once derived:
        self.log_policy = T.sum(T.log((probs*fmask) + (1. - probs)*(1. - fmask)), axis=1)
        self.log_policy = T.sum(T.log((probs**fmask)*(1. - probs)**(1. - fmask)), axis=1)
        
        if 0:
            # jacobian of the hidden repr wrt to the input
            jacobian = (probs * (1 - probs)).dimshuffle(0,'x',1) * w.dimshuffle('x',0,1)
            contractive_cost = T.sum(jacobian**2) / probs.shape[0]

        self.cost = T.mean((T.sum(probs,axis=1) - (nblocks*rate)) ** 2)
        # here we want both a single example to be sparse with a certain rate
        # but also a unit in a minibatch to be sparse with the same rate
        # so that it encourages more units being used
        self.cost += T.mean((T.mean(probs,axis=0) - rate) ** 2)

        return mask

    def baseline_cost(self, cost):
        cost = cost / self.baseline.shape[0]
        b = self.baseline
        q = (cost-b)**2
        return 0.001 * T.mean(q)
        # this is also good?
        return T.var(cost - self.baseline)

    def grad(self, costs, last_costs):
        print self.n,self.prew
        return [1./self.n_in*i for i in 
                T.Lop(f=self.log_policy, wrt=self.weights+self.prew, eval_points=costs-last_costs)]


class Model:
    def __init__(self, input, target, last_costs,
                 n_in, n_hidden, n_out,
                 rate, block_size, 
                 sparsity_function):
        self.input = input
        self.weights = []
        self.policies = []
        self.masks = []
        self.layer_outputs = []
        self.policy_class = sparsity_function().__class__.__name__
        print sparsity_function

        last_mask = None
        for i,n in enumerate(n_hidden):
            mask = None
            if i >= 0: # it can be useful to have the first hidden layer be "dense"
                policy = sparsity_function()
                mask = policy(input, last_mask, n_in, n, block_size, rate, self.weights+[],i)
                self.masks.append(mask)
                self.policies.append(policy)

            w = shared((n_in, n), 'w%d'%i)
            b = shared((n,), 'b%d'%i, 0)       
            self.weights += [w,b]

            q = sparse_dot(input, last_mask, w, mask, b, block_size)
            input = T.maximum(q, 0)
            self.layer_outputs.append(input)
            last_mask = mask
            print "sparsity:", n / block_size * rate
            n_in = n



        w = shared((n_in, n_out), 'w')
        b = shared((n_out,), 'b', 0)
        self.weights += [w,b]
        o = sparse_dot(input, last_mask, w, None, b, block_size)
        self.output = T.nnet.softmax(o)

        costs = T.sum((target - self.output)**2,axis=1)
        self.cost = T.sum(costs)


        if 0:
            # 
            L1 = T.sum([T.sum(abs(i)) for i in 
                        self.weights + [j for i in self.policies for j in i.weights]])
            L2 = T.sum([T.sum(i**2) for i in 
                        self.weights + [j for i in self.policies for j in i.weights]])
            

        # nnet grad
        self.grads = T.grad(self.cost, self.weights)
        nn_grads = self.grads
        nn_weights = self.weights

        # policy baseline grad
        baseline_cost = T.sum([i.baseline_cost(self.cost) for i in self.policies])
        baseline_grads = [j 
                          for i in self.policies if i.hasGrad
                          for j in T.grad(i.baseline_cost(self.cost), i.baseline_weights)]
        baseline_weights = [j for i in self.policies if i.hasGrad for j in i.baseline_weights]
        self.grads += baseline_grads
        self.weights += baseline_weights

        # policy grad
        self.policy_cost = T.as_tensor_variable(sum([i.cost for i in self.policies]))
        policy_grads = [0.1*j+k
                        for i in self.policies if i.hasGrad
                        for j,k in zip(i.grad(costs, last_costs),
                                       T.grad(self.policy_cost, i.weights+i.prew))]
        policy_weights = [j for i in self.policies if i.hasGrad for j in i.weights+i.prew ]
        self.weights += policy_weights
        self.grads += policy_grads
      
        # some statistics
        self.nactivations = T.cast(sum([T.sum(i) for i in self.masks]),'float32')
        self.sparsity = self.nactivations / T.sum([T.prod(i.shape) for i in self.masks])
        self.recruit_rate = T.sum([T.sum(T.sum(i, axis=0) > 0) for i in self.masks])

        lr = T.scalar()
        def merge_up(l): # add gradients on the same var
            d = {}
            for k,v in l:
                if k in d:
                    d[k] += v
                else:
                    d[k] = v
            return d.items()

        self.updates = [(i, i - lr * g) for i,g in merge_up(zip(self.weights, self.grads))]
        self.train = theano.function([self.input, target, last_costs, lr],
                                     [self.cost, costs, self.sparsity, self.nactivations, self.recruit_rate, self.policy_cost, baseline_cost],
                                     updates=self.updates)

        self.predict = theano.function([self.input],T.argmax(self.output, axis=1),profile=False)

        self.get_probs = theano.function([self.input], 
                                         T.concatenate([i.probs for i in self.policies],axis=1))

    def export_weights(self):
        return [i.get_value() for i in self.weights]

    def import_weights(self, ws):
        for i,j in zip(self.weights, ws): i.set_value(j)


def main(policy=SigmoidPolicyDropout,
         block_size=32,
         rate=0.05,
         hidden_layers=[1024,1024],
         exp_folder="/data/bengioe/condnetexp/",
         do_save_exp=False):
    print locals()
    exp_name = str(uuid.uuid4())
    exp_path = os.path.join(exp_folder, exp_name)
    print exp_path

    theano.config.compute_test_value = 'off'
    x = T.matrix('x')
    x.tag.test_value = numpy.float32(numpy.random.uniform(0,1,(2,28*28)))
    y = T.matrix('y')
    y.tag.test_value = numpy.float32(numpy.random.uniform(0,1,(2,10)))
    costs = T.vector('last costs')

    if 0: # distorted mnist
        np = numpy
        all_X = np.float32(np.load('train_inputs.npy') / 255.)
        all_Y = np.load('train_outputs.npy').reshape((all_X.shape[0],1))

        train_set = [all_X[:40000], all_Y[:40000,0]]
        valid_set = [all_X[40000:], all_Y[40000:,0]]
    elif 0: #cifar10
        trainX, trainY, testX, testY = pickle.load(file('/data/cifar/cifar_10_shuffled.pkl','r'))
        train_set = [numpy.float32(trainX/255.), trainY]
        valid_set = [numpy.float32(testX/255.), testY]
        print trainX.shape, testX.shape
    else: # mnist
        f = gzip.open("mnist.pkl.gz", 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()


    model = Model(x, y, costs, train_set[0].shape[1], hidden_layers, 10,
                  rate, block_size,
                  policy)

    batch_size = 128
    nexamples = train_set[0].shape[0]
    nminibatches = nexamples / batch_size

    best_valid_error = 100000
    best_model = None

    lr_base = 0.01
    tau = 250.0
    total_training_time = 0
    valid_errors = []

    train_costs = numpy.zeros(train_set[0].shape[0],dtype='float32')

    for epoch in range(3000):
        lr = lr_base * tau / (epoch + tau)
        t0 = time.time()
        train_cost = 0
        policy_cost = 0
        sparsity_total = 0
        baseline_total = 0
        # training epoch
        for m in range(nminibatches):
            x = train_set[0][m*batch_size:(m+1)*batch_size]
            if 0: x += numpy.random.uniform(0,0.001,x.shape)
            _y = train_set[1][m*batch_size:(m+1)*batch_size]
            y = numpy.zeros((_y.shape[0],10),'float32')
            y[numpy.arange(_y.shape[0]),_y] = 1
            c = train_costs[m*batch_size:(m+1)*batch_size]
            cost,costs,sparsity,nactivations,recruit_rate,policy_cost,baseline_cost = model.train(x,y,c,lr)
            train_costs[m*batch_size:(m+1)*batch_size] = costs
            sparsity_total += sparsity
            train_cost += cost
            baseline_total += baseline_cost
            
        train_cost /= train_set[0].shape[0]
        sparsity_total /= nminibatches
        baseline_total /= nminibatches
        t1 = time.time()
        total_training_time += t1-t0
        # validation
        cost = 0
        ps = []
        for m in range(valid_set[0].shape[0] / batch_size):
            x = valid_set[0][m*batch_size:(m+1)*batch_size]
            y = valid_set[1][m*batch_size:(m+1)*batch_size]
            p = model.predict(x)
            cost += (y != p).sum()
            if m < 8 and epoch % 10 == 0:
                p = model.get_probs(x)
                ps.append(p)

        
        # policy embeddings
        valid_error = 1.*cost / valid_set[0].shape[0]
        t2 = time.time()
        if epoch % 10 == 0:
            try:
                ps = numpy.float64(ps).reshape((-1, ps[0].shape[1]))
                tsne = TSNE(random_state=0)
                ps = tsne.fit_transform(ps)
                Y = valid_set[1][:ps.shape[0]]
                pyplot.clf()
                for i,c,s in zip(range(10),list('bgrcmyk')+[(.4,.3,.9),(.9,.4,.3),(.3,.9,.4)],'ov'*5):
                    sub = ps[Y == i]
                    pyplot.plot(sub[:,0], sub[:,1], s,color=c,ms=2,mec=c)
                pyplot.show(block=False)
                pyplot.savefig('policy_embed.png')
            except Exception,e:
                print "Couldn't make TSNE",e
            pyplot.draw()

        t3 = time.time()

        print epoch, train_cost, valid_error, 
        print "time:", t1-t0, t2-t1, t3-t2,
        print "sparsity:",sparsity_total, nactivations,
        print "recruit:",recruit_rate, "pcost",policy_cost, baseline_total,
        if numpy.isnan(train_cost) or numpy.isnan(policy_cost):
            print "\n\nnan detected, stopping"
            return

        valid_errors.append(valid_error)

        if cost < best_valid_error and do_save_exp:
            best_valid_error = cost
            print "-> best model",
            best_model = {"valid_error": valid_error,
                          "valid_errors": valid_errors,

                          "train_time": t1-t0,
                          "test_time": t2-t1,

                          "sparsity": sparsity,
                          "nactivations": nactivations,
                          "recruit_rate": recruit_rate,
                          "policy_cost": policy_cost,

                          "weights": model.export_weights(),
                          "epoch": epoch,
                          "total_training_time": total_training_time,
                          
                          "block_size": block_size,
                          "rate": rate,
                          "hidden_layers": hidden_layers,
                          "policy": model.policy_class,
                          
                          "lr": lr,
                          "batch_size": batch_size,
                          
                      }
            pickle.dump(best_model, file(exp_path,'w'), -1)
        print


if __name__ == "__main__":

    numpy.random.seed(1234)

    if 'gpu' in theano.config.device:
        srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(142857)
    else:
        srng = theano.tensor.shared_randomstreams.RandomStreams(1234)
    
    import sys
    if len(sys.argv) > 1:
        print ([SigmoidPolicyDropout, HintonDropout][int(sys.argv[1])],
               int(sys.argv[2]),
               float(sys.argv[3]),
               [int(sys.argv[4])]*2)
        main([SigmoidPolicyDropout, HintonDropout][int(sys.argv[1])],
             int(sys.argv[2]),
             float(sys.argv[3]),
             [int(sys.argv[4])]*2)
    else:
        main()
