import numpy
import scipy.misc
import gzip
import cPickle as pickle

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .sparse_dot import*


class SharedGenerator:
    def __init__(self):
        self.param_list = []
        self.param_groups = {}
    def bind(self, params, name="default"):
        if type(params)==str:
            self.param_list = self.param_groups[params]
            return
        self.param_list = params
        self.param_groups[name] = params
    def __call__(self, name, shape, init='uniform'):
        if init == "uniform":
            k = numpy.sqrt(6./numpy.sum(shape))
            values = numpy.random.uniform(-k,k,shape)
        if init == "zero":
            values = numpy.zeros(shape)
        s = theano.shared(numpy.float32(values), name=name)
        self.param_list.append(s)
        return s

    def exportToFile(self, path):
        exp = {}
        for g in self.param_groups:
            exp[g] = [i.get_value() for i in self.param_groups[g]]
        pickle.dump(exp, file(path,'w'), -1)

    def importFromFile(self, path):
        exp = pickle.load(file(path,'r'))
        for g in exp:
            for i in range(len(exp[g])):
                print g, exp[g][i].shape
                self.param_groups[g][i].set_value(exp[g][i])
shared = SharedGenerator()


class HiddenLayer:
    def __init__(self, n_in, n_out, activation, init="uniform"):
        self.W = shared("W", (n_in, n_out), init)
        self.b = shared("b", (n_out,), "zero")
        self.activation = activation

    def __call__(self, x, *args):
        return self.activation(T.dot(x,self.W) + self.b)

class InputSparseHiddenLayer:
    def __init__(self, n_in, n_out, activation, init="uniform", block_size=None):
        self.W = shared("W", (n_in, n_out), init)
        self.b = shared("b", (n_out,), "zero")
        self.activation = activation
        assert block_size != None
        self.block_size = block_size

    def __call__(self, x, xmask):
        print xmask
        return self.activation(sparse_dot(x, xmask, self.W, None, self.b, self.block_size))

class StackModel:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, *x):
        for l in self.layers:
            x = l(*x)
            if type(x) != list and type(x) != tuple:
                x = [x]
        return x



def momentum(params, grads, epsilon, lr):
    mom_ws = [theano.shared(0*(i.get_value()+1), i.name+" momentum")
              for i in params]
    mom_up = [(i, epsilon * i + (1-epsilon) * gi)
              for i,gi in zip(mom_ws, grads)]
    up = [(i, i - lr * mi) for i,mi in zip(params, mom_ws)]
    return up+mom_up


def gradient_descent(params, grads, lr):
    up = [(i, i - lr * gi) for i,gi in zip(params, grads)]
    return up


def reinforce_no_baseline(params, policy, cost, lr, regularising_cost = None):
    """
    return reinforce updates
    @policy and @cost should be of shape (minibatch_size, 1)
    @policy should be the probability of the sampled actions
    """
    log_pol = T.log(policy)
    if regularising_cost is None:
        return [(i, i - lr * gi) for i,gi in
                zip(params, T.Lop(f=log_pol, wrt=params, eval_points=cost))]
    else:
        return [(i, i - lr * (gi+gr)) for i,gi,gr in
                zip(params,
                    T.Lop(f=log_pol, wrt=params, eval_points=cost),
                    T.grad(regularising_cost, params))]


def reinforce_no_baseline_momentum(params, policy, cost, epsilon, lr, regularising_cost = None):
    """
    return reinforce updates
    @policy and @cost should be of shape (minibatch_size, 1)
    @policy should be the probability of the sampled actions
    """
    log_pol = T.log(policy)
    if regularising_cost is None:
        raise ValueError()
        return [(i, i - lr * gi) for i,gi in
                zip(params, T.Lop(f=log_pol, wrt=params, eval_points=cost))]
    else:
        return momentum(params,
                        [gi+gr
                         for gi,gr in zip(T.Lop(f=log_pol, wrt=params, eval_points=cost),
                                          T.grad(regularising_cost, params))],
                        epsilon,
                        lr)




class GenericClassificationDataset:
    def __init__(self, which, alt_path=None):
        self.alt_path = alt_path
        if which == "mnist":
            self.load_mnist()
        elif which == "cifar10":
            self.load_cifar10()
        else:
            raise ValueError("Don't know about this dataset: '%s'"%which)

    def load_mnist(self):
        f = gzip.open(self.alt_path if self.alt_path else "mnist.pkl.gz", 'rb')
        self.train,self.valid,self.test = pickle.load(f)
        f.close()
    def load_cifar10(self):
        trainX, trainY, testX, testY = pickle.load(file(self.alt_path if self.alt_path else '/data/cifar/cifar_10_shuffled.pkl','r'))
        trainX = numpy.float32(trainX / 255.)
        testX = numpy.float32(testX / 255.)
        print testX.shape, trainX.shape
        print testX.mean(),trainX.mean()
        self.train = [trainX[:40000], trainY[:40000]]
        self.valid = [trainX[40000:], trainY[40000:]]
        self.test = [testX, testY]

    def trainMinibatches(self, minibatch_size=32):
        nminibatches = self.train[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield (self.train[0][i*minibatch_size:(i+1)*minibatch_size],
                   self.train[1][i*minibatch_size:(i+1)*minibatch_size])

    def validMinibatches(self, minibatch_size=32):
        nminibatches = self.valid[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield (self.valid[0][i*minibatch_size:(i+1)*minibatch_size],
                   self.valid[1][i*minibatch_size:(i+1)*minibatch_size])

    def testMinibatches(self, minibatch_size=32):
        nminibatches = self.test[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield (self.test[0][i*minibatch_size:(i+1)*minibatch_size],
                   self.test[1][i*minibatch_size:(i+1)*minibatch_size])


    def validate(self, test, minibatch_size=32):
        cost = 0.0
        error = 0.0
        for x,y in self.validMinibatches(minibatch_size):
            c,e = test(x,y)
            cost += c
            error += e
        return (error / self.valid[0].shape[0],
                cost /  self.valid[0].shape[0])

    def doTest(self, test, minibatch_size=32):
        cost = 0.0
        error = 0.0
        for x,y in self.testMinibatches(minibatch_size):
            c,e = test(x,y)
            cost += c
            error += e
        return (error / self.test[0].shape[0],
                cost /  self.test[0].shape[0])

def get_pseudo_srqt(x):
    sqrtx = numpy.sqrt(x)
    miny = 1
    for i in range(2,x/2+1):
        if x % i == 0:
            if (sqrtx-i)**2 < (sqrtx-miny)**2:
                miny = i
    return miny



class tools:
    @staticmethod
    def export_feature_image(w, path, img_shape):
        if isinstance(w, T.sharedvar.SharedVariable):
            w = w.get_value()
        import scipy.misc as misc
        w = w.T
        ps = get_pseudo_srqt(w.shape[0])
        if len(img_shape) == 2:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1]])
        elif len(img_shape) == 3:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1], img_shape[2]])
        else:
            raise ValueError(img_shape)

        misc.imsave(path, numpy.hstack(numpy.hstack(w)))

def get_pseudo_srqt(x):
    sqrtx = numpy.sqrt(x)
    miny = 1
    for i in range(2,x/2+1):
        if x % i == 0:
            if (sqrtx-i)**2 < (sqrtx-miny)**2:
                miny = i
    return miny



class tools:
    @staticmethod
    def export_feature_image(w, path, img_shape):
        if isinstance(w, T.sharedvar.SharedVariable):
            w = w.get_value()
        import scipy.misc as misc
        w = w.T
        ps = get_pseudo_srqt(w.shape[0])
        if len(img_shape) == 2:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1]])
        elif len(img_shape) == 3:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1], img_shape[2]])
        else:
            raise ValueError(img_shape)

        misc.imsave(path, numpy.hstack(numpy.hstack(w)))

    @staticmethod
    def export_simple_plot1d(ys,path,ylabel=""):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        pyplot.plot(numpy.arange(len(ys)), ys)
        pyplot.show(block=False)
        if ylabel: pyplot.ylabel(ylabel)
        pyplot.savefig(path)

    @staticmethod
    def export_multi_plot1d(ys,path,ylabel=""):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        for i in ys:
            pyplot.plot(numpy.arange(len(i)), i)
        pyplot.show(block=False)
        if ylabel: pyplot.ylabel(ylabel)
        pyplot.savefig(path)
