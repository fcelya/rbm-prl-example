import numpy as np
import theano
import theano.tensor as tt
import theano.sparse
import json
import sys
from math import sqrt
from collections import defaultdict

def load_dataset(train_path, test_path, sep, user_based=True):
    all_users_set = set()
    all_movies_set = set()

    with open(train_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)

    tests = defaultdict(list)

    with open(test_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)
            if user_based:
                tests[uid].append((mid, float(rat)))
            else:
                tests[mid].append((uid, float(rat)))

    return list(all_users_set), list(all_movies_set), tests


def load_file(dataset, sep='::', user_based=True):
    profiles = {}
    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if user_based:
                profiles[uid].append((mid, float(rat)))
            else:
                profiles[mid].append((uid, float(rat)))
    return profiles

def cast32(x):
    return tt.cast(x, 'float32')

x = tt.matrix()
y = tt.matrix()


#CFRBM
def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]

class CFRBM:
    def __init__(self, num_visible, num_hidden, initial_v=None, initial_weigths=None, debug=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        if initial_weigths:
            initial_weights = np.load('{}.W.npy'.format(initial_weigths))
            initial_hbias = np.load('{}.h.npy'.format(initial_weigths))
            initial_vbias = np.load('{}.b.npy'.format(initial_weigths))
        else:
            initial_weights = np.array(np.random.normal(0, 0.1, size=(num_visible,num_hidden)), dtype=np.float32)
            initial_hbias = np.zeros(num_hidden, dtype=np.float32)

            if initial_v:
                initial_vbias = np.array(initial_v, dtype=np.float32)
            else:
                initial_vbias = np.zeros(num_visible, dtype=np.float32)

        self.weights = theano.shared(value=initial_weights, borrow=True, name='weights')
        self.vbias = theano.shared(value=initial_vbias, borrow=True, name='vbias')
        self.hbias = theano.shared(value=initial_hbias, borrow=True, name='hbias')

        prev_gw = np.zeros(shape=(num_visible,num_hidden), dtype=np.float32)
        self.prev_gw = theano.shared(value=prev_gw, borrow=True, name='g_w')

        prev_gh = np.zeros(num_hidden, dtype=np.float32)
        self.prev_gh = theano.shared(value=prev_gh, borrow=True, name='g_h')

        prev_gv = np.zeros(num_visible, dtype=np.float32)
        self.prev_gv = theano.shared(value=prev_gv, borrow=True, name='g_v')

        self.theano_rng = tt.shared_randomstreams.RandomStreams(np.random.RandomState(17).randint(2**30))

        if debug:
            theano.config.compute_test_value = 'warn'
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prop_up(self, vis):
        return tt.nnet.sigmoid(tt.dot(vis, self.weights) + self.hbias)

    def sample_hidden(self, vis):
        activations = self.prop_up(vis)
        h1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)
        return h1_sample, activations

    def prop_down(self, h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights.tt) + self.vbias)

    def sample_visible(self, h, k=5):
        activations = self.prop_down(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations

    def contrastive_divergence_1(self, v1):
        h1, _ = self.sample_hidden(v1)
        v2, v2a = self.sample_visible(h1)
        h2, h2a = self.sample_hidden(v2)

        return (v1, h1, v2, v2a, h2, h2a) 

    def gradient(self, v1, h1, v2, h2p, masks):
        v1h1_mask = outer(masks, h1)

        gw = ((outer(v1, h1) * v1h1_mask) - (outer(v2, h2p) * v1h1_mask)).mean(axis=0)
        gv = ((v1 * masks) - (v2 * masks)).mean(axis=0)
        gh = (h1 - h2p).mean(axis=0)

        return (gw, gv, gh)

    def cdk_fun(self, vis, masks, k=1, w_lr=0.000021, v_lr=0.000025, h_lr=0.000025, decay=0.0000, momentum=0.0):
        v1, h1, v2, v2a, h2, h2a = self.contrastive_divergence_1(vis)

        for i in range(k-1):
            v1, h1, v2, v2a, h2, h2a = self.contrastive_divergence_1(v2)

        (W, V, H) = self.gradient(v1, h1, v2, h2a, masks)

        if decay:
            W -= decay * self.weights

        updates = [
            (self.weights, cast32(self.weights + (momentum * self.prev_gw) + (W * w_lr))),
            (self.vbias, cast32(self.vbias + (momentum * self.prev_gv) + (V * v_lr))),
            (self.hbias, cast32(self.hbias + (momentum * self.prev_gh) + (H * h_lr))),
            (self.prev_gw, cast32(W)),
            (self.prev_gh, cast32(H)),
            (self.prev_gv, cast32(V))
        ]

        return theano.function([vis, masks], updates=updates)

    def predict(self, v1):
        h1, _ = self.sample_hidden(v1)
        v2, v2a = self.sample_visible(h1)
        return theano.function([v1], v2a)

    def get_weights(self):
        return self.weights, self.vbias, self.hbias

