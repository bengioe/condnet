from __future__ import print_function


import os
import time
import numpy
import theano
import theano.tensor as T
from theano_tools import*

if 'gpu' in theano.config.device:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as SRNG
else:
    from theano.tensor.shared_randomstreams import RandomStreams as SRNG
    ConvLayer.use_cudnn = False

from theano.ifelse import ifelse

import pickle as pkl

srng = SRNG(12345)


try:
    import matplotlib
    canPlot = True
except BaseException as e:
    print(e)
    canPlot = False



condnet_max_prob = theano.shared(numpy.float32(0.95))
condnet_min_prob = theano.shared(numpy.float32(0.025))

class CondConvBlock:

    def __init__(self, nin, nlayers, layer_maker, policies, sample_probabilities, layer_masks, activations):
        self.nlayers = nlayers

        self.layers = []
        self.policy_layers = []

        # collect information about activations and policy
        self.sample_probabilities = sample_probabilities
        self.layer_masks = layer_masks
        self.policies = policies
        self.activations = activations
        self.nin = nin

        for i in range(self.nlayers):
            self.layers.append(layer_maker())
            self.policy_layers.append(HiddenLayer(nin, 1, T.nnet.sigmoid))


    def __call__2(self, x):
        policy_activations = []
        h = x
        for i,l in enumerate(self.layers):
            print("layer",i)
            pol_i = self.policy_layers[i]
            p_i = pol_i(h.mean(axis=[2,3])).flatten()
            p_i = p_i * (condnet_max_prob - condnet_min_prob) + condnet_min_prob
            # act shape: (mbs, 1)
            act = T.cast(srng.uniform(p_i.shape) < p_i,'float32')
            sample_prob = p_i * act + (1-p_i)*(1-act)

            dsact = act.dimshuffle(0,'x','x','x')
            h = h * (1-dsact) + dsact * l(h)

            self.activations.append("todo")
            self.policies.append(p_i)
            self.sample_probabilities.append(sample_prob)
            self.layer_masks.append(act)
        return h

    def __call__(self, x):
        policy_activations = []
        h = x
        for i,l in enumerate(self.layers):
            print("layer",i)
            pol_i = self.policy_layers[i]
            p_i = pol_i(h.mean(axis=[2,3])).flatten()
            p_i = p_i * (condnet_max_prob - condnet_min_prob) + condnet_min_prob
            act = T.cast(srng.uniform(p_i.shape) < p_i,'float32')
            # this is really annoying but theano is doing something which I don't
            # understand and evaluates conv(h[indexes]) during the grad pass even if the
            # condition is false, so to fix this, if no example in the minibatch activates
            # a layer, I activate one randomly.
            #b = T.set_subtensor(act[T.cast(srng.uniform((1,),0,act.shape[0]), 'int32')],
            #                    numpy.int8(1))
            b = T.set_subtensor(act[T.cast(srng.uniform((1,),0,act.shape[0]), 'int32')],
                                numpy.float32(1))

            if not theano.config.device == 'cpu': # buuuut it doesn't work on cpu of course
                act = ifelse(T.eq(act.sum(),0),
                             b, act)
            # compute sample prob after "fixing" act
            sample_prob = p_i * act + (1-p_i)*(1-act)
            indexes = T.arange(x.shape[0])[act.nonzero()]

            # so I don't have to check the condition, since it should always be true:
            #h = ifelse(T.gt(h_sub.shape[0],0),
            #           T.set_subtensor(h[indexes], l(h_sub)),
            # h)

            h = T.set_subtensor(h[indexes], l(h[indexes]))

            self.activations.append("todo")
            self.policies.append(p_i)
            self.sample_probabilities.append(sample_prob)
            self.layer_masks.append(act)
        return h

one = T.as_tensor_variable(numpy.float32(1))



def plot_multiclass_activation_density(ps, path="condnet_densities.png",plotLines=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pp

    colors = [[1,0,0],[.5,.5,0],[1,0,1],
              [0,1,0],[0,1,1],
              [0,0,1],
              [.5,0,1],[1,.5,0],
              [0,.5,1],[0,1,.5]]

    pp.clf()
    fig = pp.gcf()
    ax = fig.gca()
    pp.axis([-0.1,ps[0][0].shape[0]+.1,0,1])
    ax.set_xticks(numpy.arange(0,ps[0][0].shape[0],1))
    ax.set_yticks(numpy.arange(0,1,0.1))
    for cls in range(10):
        if len(ps[cls]) < 1: continue
        nlayers = ps[cls][0].shape[0]
        xv = numpy.arange(nlayers) + cls*0.08
        nex = min(len(ps[cls]),100)
        for i in range(nex):
            if plotLines:
                pp.plot(xv,ps[cls][i],'-',ms=4.,
                        color=colors[cls]+ [numpy.sqrt(1./nex)])
            pp.plot(xv,ps[cls][i],'o',ms=4.,
                    markeredgecolor=colors[cls]+ [numpy.sqrt(1./nex)],
                    markerfacecolor=colors[cls]+ [numpy.sqrt(1./nex)],)
    for cls in range(10):
        xv = numpy.arange(nlayers) + cls*0.08
        if len(ps[cls]) < 1: continue
        pp.plot(xv, numpy.mean(ps[cls], axis=0), 'o', ms=5.,
                markerfacecolor=colors[cls],
                markeredgecolor=[0,0,0])
    pp.grid()
    pp.savefig(path)



def model_cifar_1(policies, sample_probabilities, layer_masks, activations,
                  nlayers, nfilters):


    model = StackModel([
        ConvLayer((nfilters, 3, 3, 3), lrelu(), mode='valid', normalize=True, stride=(2,2)), # 12,12
        CondConvBlock(nfilters, nlayers,
                      lambda:ConvLayer((nfilters, nfilters, 3, 3), relu, mode='half',
                                       #normalize=True,
                                       init="bengio"),
                      policies,
                      sample_probabilities,
                      layer_masks,
                      activations),
        Maxpool((2,2)), # 6,6
        #dropout(0.5),
        #ConvLayer((nfilters, nfilters, 3, 3), lrelu(), mode='valid', normalize=True), # 4,4
        ConvLayer((10, nfilters, 3, 3), lrelu(), mode='valid', normalize=True), # 2,2
        lambda x: T.mean(x, axis=[2,3]), # "mean pooling"
        HiddenLayer(10, 10, T.nnet.softmax)
        ])


    return model

def model_cifar_2(policies, sample_probabilities, layer_masks, activations,
                  nlayers, nfilters):


    model = StackModel([
        ConvLayer((96, 3, 3, 3), relu, mode='valid', stride=(2,2)), # 12,12

        #ConvLayer((96, 96, 3, 3), relu, mode='half'),
        #ConvLayer((96, 96, 3, 3), relu, mode='half'),
        #ConvLayer((96, 96, 3, 3), relu, mode='half'),
        #ConvLayer((96, 96, 3, 3), relu, mode='half'),

        ConvLayer((10, 96, 3, 3), relu, mode='valid'),
        #lambda x: T.flatten(x,2),
        lambda x: T.mean(x, axis=[2,3]), # "mean pooling"
        HiddenLayer(10, 10, T.nnet.softmax)
        ])


    return model

def model_mnist_2(policies, sample_probabilities, layer_masks, activations,
                  nlayers, nfilters):


    model = StackModel([
        ConvLayer((nfilters, 1, 5, 5), lrelu(), mode='valid', normalize=True, stride=(2,2)), # 12,12
        CondConvBlock(nfilters, nlayers,
                      lambda:ConvLayer((nfilters, nfilters, 3, 3), relu, mode='half',
                                       #normalize=True,
                                       init="bengio"),
                      policies,
                      sample_probabilities,
                      layer_masks,
                      activations),
        Maxpool((2,2)), # 6,6
        dropout(0.5),
        #ConvLayer((nfilters, nfilters, 3, 3), lrelu(), mode='valid', normalize=True), # 4,4
        ConvLayer((10, nfilters, 3, 3), lrelu(), mode='valid', normalize=True), # 2,2
        lambda x: T.mean(x, axis=[2,3]), # "mean pooling"
        HiddenLayer(10, 10, T.nnet.softmax)
        ])


    return model


def load_stats(pre='./'):
    try:
        stats = pkl.load(open(pre+'stats.pkl','r'))
    except:
        stats = {'train_errors':[],
                 'train_losses':[],
                 'valid_errors':[],
                 'valid_losses':[],
                 'current_epoch':0,
                 'train_times':[],
                 'valid_times':[],
                 }
    return stats

def save_stats(stats):
    pkl.dump(stats, open('stats.pkl','wb'), -1)

def load_params(p, path='params.pkl'):
    try:
        newp = pkl.load(open(path,'rb'))
        for p,np in zip(p,np):
            p.set_value(np)
    except:
        pass
def save_params(p, path='params.pkl'):
    pkl.dump([i.get_value() for i in p], open(path,'wb'))


def main(exp_params):
    print("Loading data")
    data = GenericClassificationDataset(exp_params['dataset'])
    image_size = {'cifar10':32,'mnist':28}[exp_params['dataset']]
    image_ndim = {'cifar10':3,'mnist':1}[exp_params['dataset']]


    print(exp_params)

    params = []
    shared.bind(params, "default")

    nlayers = int(exp_params['nlayers'])
    nfilters = int(exp_params['nfilters'])
    baseline_decay = numpy.float32(0.99)
    lambda_s = exp_params['lambda_s']
    lambda_v = exp_params['lambda_v']
    tau = exp_params['tau']

    lr = theano.shared(numpy.float32(exp_params['lr']))
    pairwise_weight = theano.shared(numpy.float32(30000.))
    pairwise_decay =  0.999



    #theano.config.compute_test_value = 'raise'
    x = T.matrix('x')
    x.tag.test_value = numpy.float32(numpy.random.random((64,image_ndim*image_size*image_size)))
    y = T.ivector('y')
    y.tag.test_value = numpy.int32([1]*64)

    policies = []
    sample_probabilities = []
    layer_masks = []
    activations = []

    print("building model")
    model = eval(exp_params['model'])(policies, sample_probabilities, layer_masks, activations,
                                      nlayers, nfilters)

    pred = model(x.reshape((x.shape[0],image_ndim,image_size,image_size)))
    # this loss just increases the variance, not good?
    #loss = T.sum(T.nnet.categorical_crossentropy(pred, y))

    losses = T.sum((pred - T.extra_ops.to_one_hot(y, 10))**2, axis=1)
    #losses = -T.log(pred[T.arange(pred.shape[0]), y])
    loss = T.sum(losses)

    _policies = policies
    if len(policies):
        policies = T.stack(policies).T # (mbsize, nlayers)
    else:
        policies = T.zeros((x.shape[0], 2))
    if len(sample_probabilities):
        sample_probabilities = T.concatenate(sample_probabilities)
    else:
        sample_probabilities = T.ones((x.shape[0], 2))
    _=abs(policies.dimshuffle(0,'x',1) - policies.dimshuffle('x',0,1))
    pairwise_cost = T.mean(1./(_+1e-6))



    baseline = theano.shared(numpy.float32(1),'baseline')

    logp = T.sum(T.log(sample_probabilities),axis=0)
    dclosses = theano.gradient.disconnected_grad(losses)

    reinforce_cost = T.sum((dclosses-baseline) * logp)
    if len(layer_masks):
        activation_ratio = T.stack(layer_masks).sum() / (nlayers * x.shape[0])
    else:
        activation_ratio = T.ones(1)

    if 0:
        reinforce_cost += (T.stack(layer_masks).sum() / (nlayers * x.shape[0]) -  tau)**2



    policy_cost = reinforce_cost

    if 1:
        #policy_cost += pairwise_weight*pairwise_cost
        if 0:
            policy_cost += lambda_s * (T.mean(abs(T.mean(policies, axis=0) - tau)) +
                                T.mean(abs(T.mean(policies, axis=1) - tau)))
        else:
            policy_cost += lambda_s * (T.mean((T.mean(policies, axis=0) - tau)**2) +
                                T.mean((T.mean(policies, axis=1) - tau)**2))
        policy_cost +=  - lambda_v * (T.mean(T.var(policies, axis=0)) +
                           T.mean(T.var(policies, axis=1)))

    cost = T.sum(loss) + policy_cost

    error = T.sum(T.neq(T.argmax(pred,axis=1), y))
    print("computing grads")
    grads = T.grad(cost, params)
    #updates = gradient_descent(params, grads, lr)
    updates = adam()(params, grads, lr)
    #updates += [(baseline, baseline_decay * baseline + (one-baseline_decay)*T.mean(dclosses))]
    #updates += [(pairwise_weight, pairwise_weight * pairwise_decay)]

    print("compiling functions")
    learn = theano.function([x,y],[loss, error, policies], updates=updates)
    test = theano.function([x,y], [error, loss, activation_ratio],
                           givens={isTestTime:one})

    # load things from possible previous runs of this experiment
    load_params(params)
    stats = load_stats()

    # create them if new
    save_stats(stats)
    save_params(params)

    rlmbs = []
    dcls = []
    activation_ratios = []

    min_valid_error = 1e64

    if "condnet_min_prob" in exp_params:
        condnet_min_prob.set_value(exp_params["condnet_min_prob"])
    if "condnet_max_prob" in exp_params:
        condnet_max_prob.set_value(exp_params["condnet_max_prob"])

    print("learning\n")
    for epoch in range(exp_params['max_epochs']):
        costs = 0
        errors = 0.0
        t0 = time.time()
        i=0
        ps = [[] for i in range(10)]

        for x,y in data.trainMinibatches(64):
            i+=1

            c,e,p = learn(x,y)

            #dcls.append(c)
            #activation_ratios.append(ar)
            costs += c
            errors += e
            if breakAfter1Epoch:
                continue

            if i < 20:
                for pi,yi in zip(p,y):
                    ps[yi].append(pi)
            if i == 20:
                asd = time.time()
                if canPlot:
                    plot_multiclass_activation_density(ps)
                    plot_multiclass_activation_density(ps, "condnet_density_lines.png", plotLines=True)
                pickle.dump(ps, open('probability_samples.pkl','wb'), -1)


        t1 = time.time()
        train_time = t1-t0
        train_error = errors / data.train[0].shape[0]

        t1 = time.time()
        verrors = 0.0
        valid_loss = 0.0
        ars = 0.0
        for x,y in data.validMinibatches(1000):
            e,l,ar = test(x,y)
            verrors += e
            valid_loss += l
            ars += ar

        ars /= data.valid[0].shape[0] / 1000


        t2 = time.time()
        valid_time = t2-t1
        valid_error = verrors / data.valid[0].shape[0]

        print()
        print(epoch, "loss", costs)
        print("  errors", errors / data.train[0].shape[0], valid_error)
        print("time:",train_time, valid_time)
        print("validation activation ratio:", ars)

        stats['train_errors'].append(train_error)
        stats['train_losses'].append(costs)
        stats['valid_errors'].append(valid_error)
        stats['valid_losses'].append(valid_loss)
        stats['train_times'].append(train_time)
        stats['valid_times'].append(valid_time)
        stats['current_epoch'] = epoch

        if breakAfter1Epoch: break


        save_stats(stats)
        save_params(params)


        if valid_error < min_valid_error:
            print("    new best validation error", valid_error, "(%s)"%min_valid_error)
            min_valid_error = valid_error
            save_params(params, "best_params.pkl")
        if canPlot:
            tools.export_multi_plot1d([stats['train_errors'], stats['valid_errors']],
                                      "errors.png", "errors", ['train','valid'])
            tools.export_multi_plot1d([stats['train_losses'], stats['valid_losses']],
                                      "loss.png", "errors", ['train', 'valid'])

    return valid_error

isTestTime = theano.shared(numpy.float32(0), 'isTestTime')
def lrelu():
    return lambda x: T.maximum(0.01*x,x) # todo

def dropout(p):
    return lambda x: ifelse(isTestTime,
                            x,
                            x * (srng.uniform(x.shape) < p) / p)



def spawn_exp_w_params(p):
    import uuid
    exp = str(uuid.uuid4())
    print(exp, p)
    s = "\n".join(["%s:%s"%(i[0],repr(i[1])) for i in p.items()])+"\n"
    open("exp/"+exp+".ini", 'w').write(s)

def parse_params(ps):
    param_strings = ps.splitlines()
    params = {}

    for i in param_strings:
        i = i.split(":")
        params[i[0]] = eval(i[1])
    return params

def spawn():

    params = {
        "dataset":"cifar10",
        "model":"model_cifar_1",
        'nlayers':4,
        'nfilters':96,
        'lambda_s':100,
        'lambda_v':10,
        'tau':0.5,
        'lr':0.003,
        'max_epochs':50,
        'condnet_min_prob': 0.1,
        'condnet_max_prob': 0.75,
    }
    if 1: spawn_exp_w_params(params)

    if 0:
        for nlayers in range(3,10):
            params['nlayers'] = nlayers
            params['tau'] = 2./nlayers
            spawn_exp_w_params(params)



def choose_and_run_exp(which):
    import os
    if which == "exp":
        exp_path = exp_name = sys.argv[3]
        if exp_name.endswith(".ini"):
            exp_path = exp_name[:-len(".ini")]
        params = parse_params(open(exp_name,'r').read())
    else:
        l = os.listdir("exp")
        l = ["exp/"+i for i in l if i.endswith('.ini')]
        found = False
        for exp_name in l:
            params = parse_params(open(exp_name,'r').read())
            exp_folder = exp_name.split('.')[0]
            exp_is_new = True
            exp_is_locked = False
            exp_is_finished = False
            exp_path = exp_folder
            if os.path.isdir(exp_path):
               exp_is_new = False
               exp_is_locked = os.path.exists(exp_path+'/lock')
               stats = load_stats(exp_path)
               if stats['current_epoch'] + 1 == params['max_epochs']:
                   exp_is_finished = True

            if which == "new" and exp_is_new:
                found = True; break
            if which == "unfinished" and (exp_is_new or not exp_is_finished) and not exp_is_locked:
                found = True; break


        if not found:
            print("Didn't find any suitable experiment (%s). Terminating."%which)
            return False

    print("Running experiment",exp_name, "mode:",which)

    pwd = os.getcwd()
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    os.chdir(exp_path)

    lock_path = os.getcwd()+"/lock"

    import atexit
    def _():
        if os.path.isfile(lock_path):
            os.remove(lock_path)
    atexit.register(_)

    open(lock_path,'w').write("")
    try:
        main(params)
    except BaseException as e:
        print("An exception occured")
        import traceback
        s = traceback.format_exc()
        print(s)
        open('exception','w').write(s)
        return False

    os.chdir(pwd)
    return True

if __name__ == "__main__":
    import sys
    breakAfter1Epoch = False


    if sys.argv[1] == "spawn":
        spawn()
    elif sys.argv[1] == "run":
        choose_and_run_exp(sys.argv[2])
    elif sys.argv[1] == "loop":
        while choose_and_run_exp(sys.argv[2]):
            pass
    elif sys.argv[1] == "debug":
        breakAfter1Epoch = True
        canPlot = False

        params = {
            "dataset":"cifar10",
            "model":"model_cifar_2",
            'nlayers':4,
            'nfilters':96,
            'lambda_s':100,
            'lambda_v':10,
            'tau':0.25,
            'lr':0.003,
            'max_epochs':200,
        }
        if 0:
            params = {
                "dataset":"cifar10",
                "model":"model_cifar_1",
                'nlayers':4,
                'nfilters':96,
                'lambda_s':100,
                'lambda_v':10,
                'tau':0.5,
                'lr':0.003,
                'max_epochs':50,
                'condnet_min_prob': 0.1,
                'condnet_max_prob': 0.75,
            }
        main(params)
    else:
        print("Don't know what to do:")
        print(sys.argv)


"""
4 layers, full,   ar 1,   79   train, 3.97 valid
2 layers, full,   ar 1,   43.5 train, 2.18 valid
0 layers, full,   ar 1,   9.37 train, 0.41 valid
 1 layer costs 0.9 seconds
 base is 0.41 seconds
4 layers, sparse, ar .46, 58.9 train, 2.85 valid
 overhead of ^ is 2.85 - (4*.46*.9+0.41) = 0.78 ~= 0.2 seconds per layer

"""
