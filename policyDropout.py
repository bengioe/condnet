import theano
import theano.tensor as T
import numpy
import uuid
import time

from theano_tools import shared, HiddenLayer, StackModel, RandomStreams, momentum,\
    GenericClassificationDataset, tools, gradient_descent, reinforce_no_baseline, \
    InputSparseHiddenLayer, reinforce_no_baseline_momentum

from theano_tools.sparse_dot import sparse_dot, sparse_dot_theano

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import cPickle as pickle
# symbolic RNG
srng = RandomStreams(142857)
from subprocess import Popen, PIPE

class PolicyDropoutLayer:
    def __init__(self, n_in, n_out, block_size, activation, do_dropout=False,
                 reinforce_params="reinforce",
                 default_params="default"):
        self.block_size = block_size
        self.nblocks = n_out / block_size
        self.do_dropout = do_dropout
        assert n_out % block_size == 0

        self.h = HiddenLayer(n_in, n_out, activation)
        shared.bind(reinforce_params)
        self.d = HiddenLayer(n_in, self.nblocks, T.nnet.sigmoid)
        shared.bind(default_params)

    def __call__(self, x, xmask=None):
        probs = self.d(x) * 0.98 + 0.01
        mask = srng.uniform(probs.shape) < probs
        print xmask
        mask.name = "mask!"
        masked = self.h.activation(sparse_dot(x, xmask, self.h.W, mask, self.h.b, self.block_size))
        if not "this is the equivalent computation in theano":
            h = self.h(x)
            if self.do_dropout:
                h = h * (srng.uniform(h.shape) < 0.5)
            h_r = h.reshape([h.shape[0], self.nblocks, self.block_size])
            masked = h_r * mask.dimshuffle(0,1,'x')
            masked = masked.reshape(h.shape)

        self.sample_probs = T.prod(mask*probs+(1-probs)*(1-mask), axis=1)
        self.probs = probs
        return masked, mask



def build_model(new_model=True):
    momentum_epsilon = 0.9

    block_size = 64
    nblocks = [8,8,8]
    rate = [1/4.,1/32.,1/32.]
    L2reg = 0.05

    lambda_b = [80,40,40]
    lambda_v = [10,20,20]
    learning_rates = [0.01,0.5,0.5]


    do_dropout = False

    print locals()

    hyperparams = locals()




    if new_model:
        expid = str(uuid.uuid4())
        import os
        import os.path
        code = file(os.path.abspath(__file__),'r').read()
        os.mkdir(expid)
        os.chdir(expid)
        file('code.py','w').write(code)

        print expid

        f = file("params.txt",'w')
        for i in hyperparams:
            f.write("%s:%s\n"%(i,str(hyperparams[i])))
        f.close()


    params = []
    reinforce_params = []
    shared.bind(reinforce_params, "reinforce")
    shared.bind(params)

    rect = lambda x:T.maximum(0,x)
    act = T.tanh

    model = StackModel([PolicyDropoutLayer(32*32*3, block_size*nblocks[0],
                                           block_size, act),
                        PolicyDropoutLayer(block_size*nblocks[0], block_size*nblocks[1],
                                           block_size, act),
                        PolicyDropoutLayer(block_size*nblocks[1], block_size*nblocks[2],
                                           block_size, act),
                        InputSparseHiddenLayer(block_size*nblocks[-1], 10, T.nnet.softmax,
                                               block_size=block_size)])


    x = T.matrix()
    y = T.ivector()
    lr = T.scalar()

    y_hat, = model(x)
    loss = T.nnet.categorical_crossentropy(y_hat, y)
    cost = T.sum(loss)
    l2 = lambda x:sum([T.sum(i**2) for i in x])
    updates = []
    all_probs = []
    assymetric_distance = lambda x: T.minimum(0,x) * -0.1 + T.maximum(0,x)
    for i in range(len(model.layers)-1):
        probs = model.layers[i].probs
        sample_probs = model.layers[i].sample_probs
        layer_params = [model.layers[i].d.W, model.layers[i].d.b]
        all_probs.append(probs)

        l2_batchwise = lambda_b[i] * T.sum(abs(T.mean(probs, axis=0) - rate[i])**2)
        l2_exawise   = lambda_b[i] * 0.001*T.sum(abs(T.mean(probs, axis=1) - rate[i])**2)
        batch_var    = lambda_v[i] * T.sum(T.var(probs, axis=0))
        batch_var    += lambda_v[i] * 0.1*T.sum(T.var(probs, axis=1))

        #l2_batchwise = lambda_b[i] * T.sum(assymetric_distance(T.mean(probs, axis=0) - rate[i]))
        #l2_exawise = lambda_b[i] * 0.001*T.sum(assymetric_distance(T.mean(probs, axis=1) - rate[i]))
        #batch_var   = lambda_v[i] * T.sum(T.mean(assymetric_distance(rate-probs), axis=0))
        #batch_var =  T.sum(T.mean((probs-(1-rate))**2, axis=0))
        regularising_cost = l2_batchwise + l2_exawise - batch_var + L2reg * l2(layer_params)
        updates += reinforce_no_baseline(layer_params, sample_probs,
                                                  loss-loss.min(),# momentum_epsilon,
                                                  lr*learning_rates[i],
                                                  regularising_cost)

    error = T.sum(T.neq(y_hat.argmax(axis=1), y))
    nn_regularization = L2reg * l2(params)

    grads = T.grad(cost + nn_regularization, params)
    updates += gradient_descent(params, grads, lr)
    print params, reinforce_params

    learn = theano.function([x,y,lr], [cost, error, T.concatenate(all_probs,axis=1)], updates=updates, allow_input_downcast=True)
    test = theano.function([x,y], [cost, error], allow_input_downcast=True)

    return model,learn,test

def main():

    data = GenericClassificationDataset("cifar10", "cifar_10_shuffled.pkl")
    N = data.train[0].shape[0] * 1.

    model, learn, test = build_model()

    do_video = True
    some_probs = []
    fps = 30
    try:
        video = Popen(['avconv', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg',#'rawvideo', "-pix_fmt", "rgba",
                       '-r', str(fps),'-s','800x600', '-i', '-',
                       '-qscale', '9', '-r', str(fps), 'video.webm'], stdin=PIPE)
    except Exception,e:
        print "Cannot do video:",e
        do_video = False



    epoch = 0
    def plot_mean_activation_and_stuff(some_probs, Y, do_tsne=False):
        pyplot.clf()
        probs = numpy.float32(some_probs)
        xv = numpy.arange(probs.shape[1])#probs.var(axis=0)
        yv = probs.mean(axis=0)
        pyplot.axis([-0.1, probs.shape[1],0,1])
        for k in range(probs.shape[1]):
            pyplot.plot(xv[k]*numpy.ones(probs.shape[0]),probs[:,k],'o',ms=4.,
                        markeredgecolor=(1, 0, 0, 0.01),
                        markerfacecolor=(1, 0, 0, 0.01),)
        pyplot.plot(xv,yv, 'bo')
        pyplot.show(block=False)
        if do_video:
            pyplot.savefig(video.stdin, format='jpeg')
            video.stdin.flush()
        pyplot.savefig('epoch_probs.png')

        if not do_tsne: return
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(random_state=0)
            ps = tsne.fit_transform(numpy.float64(probs[:400]))
            pyplot.clf()
            Y = numpy.int32(Y)[:400]
            for i,c,s in zip(range(10),list('bgrcmyk')+[(.4,.3,.9),(.9,.4,.3),(.3,.9,.4)],'ov'*5):
                sub = ps[Y == i]
                pyplot.plot(sub[:,0], sub[:,1], s,color=c,ms=3,mec=c)
            pyplot.show(block=False)
            pyplot.savefig('probs_embed.png')
        except ImportError:
            print "cant do tsne"



    experiment = {"results":None,
                  }

    lr = 0.001 # * 100 / (i+100)
    costs = []
    errors = []
    valid_costs = []
    valid_errors = []
    for i in range(1000):
        epoch = i
        cost = 0
        error = 0
        probs = 0
        some_probs = []
        ys = []
        do_tsne = True
        for x,y in data.trainMinibatches(128):
            c,e,p = learn(x,y,lr)
            cost += c
            error += e
            probs += p.sum(axis=0)
            if len(some_probs) < 1000:
                some_probs += list(p)
                ys += list(y)
            else:
                plot_mean_activation_and_stuff(some_probs, ys, do_tsne)
                some_probs = []
                do_tsne = False

        t0 = time.time()
        valid_error, valid_cost = data.validate(test, 50)
        valid_time = time.time() - t0
        print
        print i, cost/N, error/N
        print valid_error, valid_cost, valid_time
        print probs.mean() / N
        print probs / N
        errors.append(error/N)
        costs.append(cost/N)
        valid_errors.append(valid_error)
        valid_costs.append(valid_cost)
        tools.export_feature_image(model.layers[0].h.W, "W_img.png", (32,32,3))
        tools.export_feature_image(model.layers[0].d.W, "Z_img.png", (32,32,3))
        tools.export_multi_plot1d([errors, valid_errors], "errors.png", "error")
        tools.export_multi_plot1d([costs, valid_costs], "costs.png", "cost")
        experiment["results"] = [valid_costs, valid_errors, costs, errors]
        experiment["valid_time"] = valid_time
        pickle.dump(experiment, file("experiment.pkl",'w'),-1)
        shared.exportToFile("weights.pkl")
    video.stdin.close()
    video.wait()


def test(expid):
    # to test:
    # OMP_NUM_THREADS=1 THEANO_FLAGS=device=cpu taskset -c 0 python $(expip)/code.py $(expid)
    import os
    os.chdir(expid)
    print "loading data"
    data = GenericClassificationDataset("cifar10", "../cifar_10_shuffled.pkl")

    global sparse_dot
    print "building model"
    model,learn,test = build_model(False)
    print "importing weights"
    shared.importFromFile("weights.pkl")
    print "testing"
    import time
    t0 = time.time()
    test_error, test_cost = data.doTest(test, 50)
    t1 = time.time()
    print "Error, cost, time(s)"
    print test_error, test_cost, t1-t0
    specialized_test_time = t1-t0

    sparse_dot = sparse_dot_theano
    print "building model"
    model,learn,test = build_model(False)
    print "importing weights"
    shared.importFromFile("weights.pkl")
    print "testing"
    import time
    t0 = time.time()
    test_error, test_cost = data.doTest(test, 50)
    t1 = time.time()
    normal_test_time = t1-t0

    print "Error, cost, time(s)"
    print test_error, test_cost, t1-t0

    f= file("test_results.txt",'w')
    f.write("specialized:%f\ntheano:%f\nerror:%f\n"%(specialized_test_time, normal_test_time, test_error))
    f.close()

if __name__ == "__main__":
    import sys
    print sys.argv
    if len(sys.argv) <= 1:
        main()
    else:
        test(sys.argv[1])
