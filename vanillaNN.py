import theano
import theano.tensor as T
import numpy
import uuid
import time

from theano_tools import shared, HiddenLayer, StackModel, RandomStreams, momentum,\
    GenericClassificationDataset, tools, gradient_descent, reinforce_no_baseline, \
    InputSparseHiddenLayer, reinforce_no_baseline_momentum

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import cPickle as pickle
# symbolic RNG
srng = RandomStreams(142857)


class Dropout:
    def __init__(self):
        pass
    def __call__(self, x):
        return (srng.uniform(x.shape) < 0.5) * x

def build_model(new_model=True):
    momentum_epsilon = 0.9

    nhidden = [64,64]
    L2reg = 0.001

    vanilla = True

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
    shared.bind(params)

    rect = lambda x:T.maximum(0,x)
    act = T.tanh

    model = StackModel([HiddenLayer(32*32*3, nhidden[0],  act),
                        Dropout(),
                        HiddenLayer(nhidden[0], nhidden[1], act),
                        Dropout(),
                        HiddenLayer(nhidden[-1], 10, T.nnet.softmax)])


    x = T.matrix()
    y = T.ivector()
    lr = T.scalar()

    y_hat, = model(x)
    loss = T.nnet.categorical_crossentropy(y_hat, y)
    cost = T.sum(loss)
    l2 = lambda x:sum([T.sum(i**2) for i in x])
    updates = []
    error = T.sum(T.neq(y_hat.argmax(axis=1), y))
    nn_regularization = L2reg * l2(params)

    grads = T.grad(cost + nn_regularization, params)
    updates += gradient_descent(params, grads, lr)

    learn = theano.function([x,y,lr], [cost, error], updates=updates, allow_input_downcast=True)
    test = theano.function([x,y], [cost, error], allow_input_downcast=True)

    return model,learn,test

def main():

    data = GenericClassificationDataset("cifar10", "cifar_10_shuffled.pkl")
    N = data.train[0].shape[0] * 1.

    model, learn, test = build_model()


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
        for x,y in data.trainMinibatches(128):
            c,e = learn(x,y,lr)
            cost += c
            error += e

        t0 = time.time()
        valid_error, valid_cost = data.validate(test, 50)
        valid_time = time.time() - t0
        print
        print i, cost/N, error/N
        print valid_error, valid_cost, valid_time
        errors.append(error/N)
        costs.append(cost/N)
        valid_errors.append(valid_error)
        valid_costs.append(valid_cost)
        tools.export_feature_image(model.layers[0].W, "W_img.png", (32,32,3))
        tools.export_multi_plot1d([errors, valid_errors], "errors.png", "error")
        tools.export_multi_plot1d([costs, valid_costs], "costs.png", "cost")
        experiment["results"] = [valid_costs, valid_errors, costs, errors]
        experiment["valid_time"] = valid_time
        pickle.dump(experiment, file("experiment.pkl",'w'),-1)
        shared.exportToFile("weights.pkl")
    #video.stdin.close()
    #video.wait()


def test(expid):
    # to test:
    # OMP_NUM_THREADS=1 THEANO_FLAGS=device=cpu taskset -c 0 python $(expip)/code.py $(expid)
    import os
    os.chdir(expid)
    print "loading data"
    data = GenericClassificationDataset("cifar10", "../cifar_10_shuffled.pkl")

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
    normal_test_time = t1-t0

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
