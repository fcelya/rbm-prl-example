import numpy as np
import theano
import theano.tensor as tt
import theano.sparse
import json
import sys
import csv
import time
from math import sqrt
from collections import defaultdict 
from sklearn.metrics import precision_recall_fscore_support

iteration_str = "\nEnd iter {} - k/lr: {}/{} momentum: {} - MAE/RMSE: {}/{}"

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def avg(x):
    return sum(x)/len(x)


def _expand_line(line, k=5):
    expanded = [0] * (len(line) * k)
    for i, el in enumerate(line):
        if float(el) != 0.:
            el = float(el)
            expanded[(i*k) + int(round(el)) - 1] = 1
    return expanded


def expand(data, k=5):
    new = []
    for m in data:
        new.extend(_expand_line(m.tolist(), k))

    return np.array(new).reshape(data.shape[0], data.shape[1] * k)


def revert_expected_value(m, k=5, do_round=True):
    mask = list(range(1, k+1))
    vround = np.vectorize(round)

    if do_round:
        users = vround((m.reshape(-1, k) * mask).sum(axis=1))
    else:
        users = (m.reshape(-1, k) * mask).sum(axis=1)

    return np.array(users).reshape(m.shape[0], m.shape[1] / k)

def cast32(x):
    return tt.cast(x, 'float32')

def read_experiment(path):

    with open(path) as data:
        experiments = json.loads(data.read())['experiments']

    return experiments

def load_file(dataset, sep ='::', user_based=True):
    profiles = {}
    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            user, movie, rating, timstamp = line.strip.split(sep)
            if user_based:
                profiles[user].append((movie),float(rating))
            else:
                profiles[movie].append((user),float(rating))
    return profiles



def load_data(train_path, test_path, sep, user_based = True):
    users_set = set()
    movies_set = set()
    jobs_set = ['administrator', 'executive', 'retired', 'lawyer',' entertainment', 'marketing','writer','none','scientist','healthcare','other','student','educator','technician', 'librarian','programmer','artist','salesman','doctor','homemaker','engineer']
    sex_set = ['M', 'F']
    ages_set = ['1','18','25','35','45','50','56']
    genres_set = ['unknown','Action','Adventure','Animation', 'Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

    #open training path
    with open(train_path, 'rt') as data:
        for i, line in enumerate(data):
            user, movie, rating, timstamp = line.strip().split(sep)
            if user not in users_set:
                users_set.add(user)
            if movie not in movies_set:
                movies_set.add(movie)

    tests = defaultdict(list)

    with open(test_path, 'rt') as data:
        for i, line in enumerate(data):
            user, movie, rating, timstamp = line.strip.split(sep)
            if user not in users_set:
                users_set.add(user)
            if movie not in movies_set:
                movies_set.add(movie)
            if user_based:
                tests[user].append((movie), float(rating))
            else:
                tests[movie].append((user), float(rating))

    return list(users_set), list(movies_set), list(jobs_set), list(sex_set), list(ages_set), list(genres_set)


x = tt.matrix()
y = tt.matrix()


#CFRBM
def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]

class CFRBM:
    def __init__(self, num_visible_x, num_visible_o, num_visible_s, num_visible_a, num_visible_g, num_hidden, initial_v_x=None, initial_v_o=None, initial_v_s=None, initial_v_a=None, initial_v_g=None, initial_weights_x=None, initial_weights_o=None, initial_weights_s=None, initial_weights_a=None, initial_weights_g=None, debug=False):
        self.dim_x = (num_visible_x, num_hidden)
        self.dim_o = (num_visible_o, num_hidden)
        self.dim_s = (num_visible_s, num_hidden)
        self.dim_a = (num_visible_a, num_hidden)
        self.dim_g = (num_visible_g, num_hidden)
        self.num_visible_x = num_visible_x
        self.num_visible_o = num_visible_o
        self.num_visible_s = num_visible_s
        self.num_visible_a = num_visible_a
        self.num_visible_g = num_visible_g
        self.num_hidden = num_hidden

        if initial_weights_x:
            initial_weights_x = np.load('{}.W_x.npy'.format(initial_weights_x))
            initial_weights_o = np.load('{}.W_o.npy'.format(initial_weights_o))
            initial_weights_s = np.load('{}.W_s.npy'.format(initial_weights_s))
            initial_weights_a = np.load('{}.W_a.npy'.format(initial_weights_a))
            initial_weights_g = np.load('{}.W_g.npy'.format(initial_weights_g))
            initial_hbias = np.load('{}.h.npy'.format(initial_weights_x))
            initial_vbias_x = np.load('{}.b_x.npy'.format(initial_weights_x))
            initial_vbias_o = np.load('{}.b_o.npy'.format(initial_weights_o))
            initial_vbias_s = np.load('{}.b_s.npy'.format(initial_weights_s))
            initial_vbias_a = np.load('{}.b_a.npy'.format(initial_weights_a))
            initial_vbias_g = np.load('{}.b_g.npy'.format(initial_weights_g))
        else:
            initial_weights_x = np.array(np.random.normal(0, 0.1, size=self.dim_x),dtype=np.float32)
            initial_weights_o = np.array(np.random.normal(0, 0.1, size=self.dim_o), dtype=np.float32)
            initial_weights_s = np.array(np.random.normal(0, 0.1, size=self.dim_s), dtype=np.float32)
            initial_weights_a = np.array(np.random.normal(0, 0.1, size=self.dim_a), dtype=np.float32)
            initial_weights_g = np.array(np.random.normal(0, 0.1, size=self.dim_g), dtype=np.float32)
            initial_hbias = np.zeros(num_hidden, dtype=np.float32)

            if initial_v_x:
                initial_vbias_x = np.array(initial_v_x, dtype=np.float32)
                initial_vbias_o = np.array(initial_v_o, dtype=np.float32)
                initial_vbias_s = np.array(initial_v_s, dtype=np.float32)
                initial_vbias_a = np.array(initial_v_a, dtype=np.float32)
                initial_vbias_g = np.array(initial_v_g, dtype=np.float32)
            else:
                initial_vbias_x = np.zeros(num_visible_x, dtype=np.float32)
                initial_vbias_o = np.zeros(num_visible_o, dtype=np.float32)
                initial_vbias_s = np.zeros(num_visible_s, dtype=np.float32)
                initial_vbias_a = np.zeros(num_visible_a, dtype=np.float32)
                initial_vbias_g = np.zeros(num_visible_g, dtype=np.float32)


        # weights 
        self.weights_x = theano.shared(value=initial_weights_x, borrow=True, name='weights_x')
        self.weights_o = theano.shared(value=initial_weights_o, borrow=True, name='weights_o')
        self.weights_s = theano.shared(value=initial_weights_s, borrow=True, name='weights_s')
        self.weights_a = theano.shared(value=initial_weights_a, borrow=True, name='weights_a')
        self.weights_g = theano.shared(value=initial_weights_g, borrow=True, name='weights_g')
        
        
        # biases
        self.vbias_x = theano.shared(value=initial_vbias_x, borrow=True, name='vbias_x')
        self.vbias_o = theano.shared(value=initial_vbias_o, borrow=True, name='vbias_o')
        self.vbias_s = theano.shared(value=initial_vbias_s, borrow=True, name='vbias_s')
        self.vbias_a = theano.shared(value=initial_vbias_a, borrow=True, name='vbias_a')
        self.vbias_g = theano.shared(value=initial_vbias_g, borrow=True, name='vbias_g')
        self.hbias = theano.shared(value=initial_hbias, borrow=True, name='hbias')

        prev_gw_x = np.zeros(shape=self.dim_x, dtype=np.float32)
        self.prev_gw_x = theano.shared(value=prev_gw_x, borrow=True, name='g_w_x')

        prev_gw_o = np.zeros(shape=self.dim_o, dtype=np.float32)
        self.prev_gw_o = theano.shared(value=prev_gw_o, borrow=True, name='g_w_o')

        prev_gw_s = np.zeros(shape=self.dim_s, dtype=np.float32)
        self.prev_gw_s = theano.shared(value=prev_gw_s, borrow=True, name='g_w_s')

        prev_gw_a = np.zeros(shape=self.dim_a, dtype=np.float32)
        self.prev_gw_a = theano.shared(value=prev_gw_a, borrow=True, name='g_w_a')

        prev_gw_g = np.zeros(shape=self.dim_g, dtype=np.float32)
        self.prev_gw_g = theano.shared(value=prev_gw_g, borrow=True, name='g_w_g')

        prev_gh = np.zeros(num_hidden, dtype=np.float32)
        self.prev_gh = theano.shared(value=prev_gh, borrow=True, name='g_h')

        prev_gv_x = np.zeros(num_visible_x, dtype=np.float32)
        self.prev_gv_x = theano.shared(value=prev_gv_x, borrow=True, name='g_v_x')

        prev_gv_o = np.zeros(num_visible_o, dtype=np.float32)
        self.prev_gv_o = theano.shared(value=prev_gv_o, borrow=True, name='g_v_o')

        prev_gv_s = np.zeros(num_visible_s, dtype=np.float32)
        self.prev_gv_s = theano.shared(value=prev_gv_s, borrow=True, name='g_v_s')

        prev_gv_a = np.zeros(num_visible_a, dtype=np.float32)
        self.prev_gv_a = theano.shared(value=prev_gv_a, borrow=True, name='g_v_a')

        prev_gv_g = np.zeros(num_visible_g, dtype=np.float32)
        self.prev_gv_g = theano.shared(value=prev_gv_g, borrow=True, name='g_v_g')

        self.theano_rng = tt.shared_randomstreams.RandomStreams(
            np.random.RandomState(17).randint(2**30))
        
        if debug:
            theano.config.compute_test_value = 'warn'
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prop_up(self, vis, occ, sex, age, genre):
        return tt.nnet.sigmoid(tt.dot(vis, self.weights_x) + tt.dot(occ, self.weights_o)  + tt.dot(sex, self.weights_s) + tt.dot(age, self.weights_a) + tt.dot(genre, self.weights_g) + self.hbias)

    def sample_hidden(self, vis, occ, sex, age, genre):
        activations = self.prop_up(vis, occ, sex, age, genre)
        h1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)
        return h1_sample, activations
    
    def prop_down_x(self, h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_x.tt) + self.vbias)

    def drop_down_o(self,h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_o.tt) + self.vbias)

    def drop_down_s(self,h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_s.tt) + self.vbias)

    def drop_down_a(self,h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_a.tt) + self.vbias)
        
    def drop_down_g(self,h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_g.tt) + self.vbias)
        

    def sample_visible_x(self, h, k=5):
        activations = self.prop_down_x(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations

    def sample_visible_o(self, h, k=5):
        activations = self.prop_down_o(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations

    def sample_visible_s(self, h, k=5):
        activations = self.prop_down_s(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations

    def sample_visible_a(self, h, k=5):
        activations = self.prop_down_a(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations
    
    def sample_visible_g(self, h, k=5):
        activations = self.prop_down_g(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations = activations / partitions.reshape(activations.shape)
        v1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample, activations


    def contrastive_divergence_1(self, v1_x, v1_o, v1_s, v1_a, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_o, v1_s, v1_a, v1_g)
        v2_x, v2a_x = self.sample_visible_x(h1)
        v2_o, v2a_o = self.sample_visible_o(h1)
        v2_s, v2a_s = self.sample_visible_s(h1)
        v2_a, v2a_a = self.sample_visible_a(h1)
        v2_g, v2a_g = self.sample_visible_g(h1)
        h2, h2a = self.sample_hidden(v2_x, v2_o, v2_s, v2_a, v2_g)

        return (v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a)

    def gradient(self, v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2_o, v2_s, v2_a, v2_g, h2p, masks_x, masks_o, masks_s, masks_a, masks_g):
        v1_xh1_mask_x = outer(masks_x, h1)
        v1_oh1_mask_o = outer(masks_o, h1)
        v1_sh1_mask_s = outer(masks_s, h1)
        v1_ah1_mask_a = outer(masks_a, h1)
        v1_ah1_mask_g = outer(masks_g, h1)

        gw_x = ((outer(v1_x, h1) * v1_xh1_mask_x) - (outer(v2_x, h2p) * v1_xh1_mask_x)).mean(axis=0)
        gw_o = ((outer(v1_o, h1) * v1_oh1_mask_o) - (outer(v2_o, h2p) * v1_oh1_mask_o)).mean(axis=0)
        gw_s = ((outer(v1_s, h1) * v1_sh1_mask_s) - (outer(v2_s, h2p) * v1_sh1_mask_s)).mean(axis=0)
        gw_a = ((outer(v1_a, h1) * v1_ah1_mask_a) - (outer(v2_a, h2p) * v1_ah1_mask_a)).mean(axis=0)
        gw_g = ((outer(v1_g, h1) * v1_ah1_mask_g) - (outer(v2_g, h2p) * v1_ah1_mask_g)).mean(axis=0)
        gv_x = ((v1_x * masks_x) - (v2_x * masks_x)).mean(axis=0)
        gv_o = ((v1_o * masks_o) - (v2_o * masks_o)).mean(axis=0)
        gv_s = ((v1_s * masks_s) - (v2_s * masks_s)).mean(axis=0)
        gv_a = ((v1_a * masks_a) - (v2_a * masks_a)).mean(axis=0)
        gv_g = ((v1_g * masks_g) - (v2_g * masks_g)).mean(axis=0)
        gh = (h1 - h2p).mean(axis=0)

        return (gw_x, gw_o, gw_s, gw_a, gw_g, gv_x, gv_o, gv_s, gv_a, gv_g, gh)

    def cdk_fun(self, vis_x, vis_o, vis_s, vis_a, vis_g, masks_x, masks_o, masks_s, masks_a, masks_g, k=1, w_lr=0.000021, v_lr=0.000025, h_lr=0.000025, decay=0.0000, momentum=0.0):
        v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a = self.contrastive_divergence_1(vis_x, vis_o, vis_s, vis_a, vis_g)

        for i in range(k-1):
            v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a = self.contrastive_divergence_1(v2_x, v2_o, v2_s, v2_a, v2_g)

        (W_x, W_o, W_s, W_a, W_g, V_x, V_o, V_s, V_a, V_g, H) = self.gradient(v1_x, v1_o, v1_s, v1_a, v1_g, h1,  v2_x, v2_o, v2_s, v2_a, v2_g, h2a, masks_x, masks_o, masks_s, masks_a, masks_g)

        if decay:
            W_x -= decay * self.weights_x
            W_o -= decay * self.weights_o
            W_s -= decay * self.weights_s
            W_a -= decay * self.weights_a
            W_g -= decay * self.weights_g

        updates = [
            (self.weights_x, cast32(self.weights_x + (momentum * self.prev_gw_x) + (W_x * w_lr))),
            (self.weights_o, cast32(self.weights_o + (momentum * self.prev_gw_o) + (W_o * w_lr))),
            (self.weights_s, cast32(self.weights_s + (momentum * self.prev_gw_s) + (W_s * w_lr))),
            (self.weights_a, cast32(self.weights_a + (momentum * self.prev_gw_a) + (W_a * w_lr))),
            (self.weights_g, cast32(self.weights_g + (momentum * self.prev_gw_g) + (W_g * w_lr))),
            (self.vbias_x, cast32(self.vbias_x + (momentum * self.prev_gv_x) + (V_x * v_lr))),
            (self.vbias_o, cast32(self.vbias_o + (momentum * self.prev_gv_o) + (V_o * v_lr))),
            (self.vbias_s, cast32(self.vbias_s + (momentum * self.prev_gv_s) + (V_s * v_lr))),
            (self.vbias_a, cast32(self.vbias_a + (momentum * self.prev_gv_a) + (V_a * v_lr))),
            (self.vbias_g, cast32(self.vbias_g + (momentum * self.prev_gv_g) + (V_g * v_lr))),
            (self.hbias, cast32(self.hbias + (momentum * self.prev_gh) + (H * h_lr))),
           
           
            (self.prev_gw_x, cast32(W_x)),
            (self.prev_gw_o, cast32(W_o)),
            (self.prev_gw_s, cast32(W_s)),
            (self.prev_gw_a, cast32(W_a)),
            (self.prev_gw_g, cast32(W_g)),
            (self.prev_gh, cast32(H)),
            (self.prev_gv_x, cast32(V_x)),
            (self.prev_gv_o, cast32(V_o)),
            (self.prev_gv_s, cast32(V_s)),
            (self.prev_gv_a, cast32(V_a)),
            (self.prev_gv_g, cast32(V_g))
        ]

        return theano.function([[vis_x, vis_o, vis_s, vis_a, vis_g, masks_x, masks_o, masks_s, masks_a, masks_g]], updates=updates)

    def predict(self, v1_x, v1_o, v1_s, v1_a, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_o, v1_s, v1_a, v1_g)
        v2, v2a = self.sample_visible(h1)

    def get_weights(self):
        return self.weights, self.vbias, self.hbias

def run(name, dataset, user_info, movie_info, config, all_users, all_movies, all_occupations, all_sex, all_ages, all_genres, tests, initial_v, sep):
    config_name = config['name']
    number_hidden = config['number_hidden']
    epochs = config['epochs']
    ks = config['ks']
    momentums = config['momentums']
    l_w = config['l_w']
    l_v = config['l_v']
    l_h = config['l_h']
    decay = config['decay']
    batch_size = config['batch_size']

    config_result = config.copy()
    config_result['results'] = []

    vis_x = tt.matrix()
    vis_o = tt.matrix()
    vis_s = tt.matrix()
    vis_a = tt.matrix()
    vis_g = tt.matrix()
    vmasks_x = tt.matrix()
    vmasks_o = tt.matrix()
    vmasks_s = tt.matrix()
    vmasks_a = tt.matrix()
    vmasks_g = tt.matrix()

    rbm = CFRBM(len(all_movies) * 5, len(all_occupations), 1, len(all_ages), len(all_genres), number_hidden)

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[uid].append((mid, float(rat)))

    print("Users and ratings loaded")

    user_occ = defaultdict(list)
    user_sex = defaultdict(list)
    user_age = defaultdict(list)

    r = csv.reader(open(user_info, 'rb'), delimiter='|')
    for row in r:
        user_age[row[0]] = [int(x) for x in row[1:7]]
        user_sex[row[0]] = [int(row[7])]
        user_occ[row[0]] = [int(x) for x in row[8:]]

    print("User info loaded")

    movie_genres = [0.] * len(all_movies) * len(all_genres)

    r = csv.reader(open(movie_info, 'rb'), delimiter='|')
    for row in r:
        for _i in range(len(all_genres)):
        #    movie_genres[len(all_genres) * all_movies.index(movie_id) + _i] = int(row[5+_i])
            pass

    print("Movie genres loaded")

    for j in range(epochs):
        def get_index(col):
            if j/(epochs/len(col)) < len(col):
                return j/(epochs/len(col))
            else:
                return -1

        index = get_index(ks)
        mindex = get_index(momentums)
        icurrent_l_w = get_index(l_w)
        icurrent_l_v = get_index(l_v)
        icurrent_l_h = get_index(l_h)

        k = ks[index]
        momentum = momentums[mindex]
        current_l_w = l_w[icurrent_l_w]
        current_l_v = l_v[icurrent_l_v]
        current_l_h = l_h[icurrent_l_h]

        train = rbm.cdk_fun(vis_x, vis_o, vis_s, vis_a, vis_g, vmasks_x, vmasks_o, vmasks_s, vmasks_a, vmasks_g, k=k, w_lr=current_l_w, v_lr=current_l_v, h_lr=current_l_h, decay=decay,momentum=momentum)
        predict = rbm.predict(vis_x, vis_o, vis_s, vis_a, vis_g)

        start_time = time.time()

        for batch_i, batch in enumerate(chunker(profiles.keys(), batch_size)):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            occ_profiles = {}
            sex_profiles = {}
            age_profiles = {}
            genre_profiles = {}
            masks_x = {}
            masks_o = {}
            masks_s = {}
            masks_a = {}
            masks_g = {}

            for userid in batch:
                user_profile = [0.] * len(all_movies)
                occ_profile = [0.] * len(all_occupations)
                sex_profile = [0.] * 1
                age_profile = [0.] * len(all_ages)

                mask_x = [0] * (len(all_movies) * 5)
                mask_o = [1] * (len(all_occupations))
                mask_s = [1] * (1)
                mask_a = [1] * (len(all_ages))
                mask_g = [0] * (len(all_movies) * len(all_genres))

                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(5):
                        mask_x[5 * all_movies.index(movie_id) + _i] = 1
                    for _i in range(len(all_genres)):
                        mask_g[len(all_genres) * all_movies.index(movie_id) + _i] = 1

                mask_o = [1] * len(all_occupations)
                mask_s = [1] * 1
                mask_a = [1] * len(all_ages)

                example_x = expand(np.array([user_profile])).astype('float32')
                example_o = expand(np.array([occ_profile]), k=1).astype('float32')
                example_s = expand(np.array([sex_profile]), k=1).astype('float32')
                example_a = expand(np.array([age_profile]), k=1).astype('float32')
                example_g = np.array(movie_genres).astype('float32')

                bin_profiles[userid] = example_x
                occ_profiles[userid] = example_o
                sex_profiles[userid] = example_s
                age_profiles[userid] = example_a
                genre_profiles[userid] = example_g
                masks_x[userid] = mask_x
                masks_o[userid] = mask_o
                masks_s[userid] = mask_s
                masks_a[userid] = mask_a
                masks_g[userid] = mask_g

            profile_batch = [bin_profiles[id] for id in batch]
            occ_batch = [occ_profiles[id] for id in batch]
            sex_batch = [sex_profiles[id] for id in batch]
            age_batch = [age_profiles[id] for id in batch]
            genre_batch = [genre_profiles[id] for id in batch]

            masks_x_batch = [masks_x[id] for id in batch]
            masks_o_batch = [masks_o[id] for id in batch]
            masks_s_batch = [masks_s[id] for id in batch]
            masks_a_batch = [masks_a[id] for id in batch]
            masks_g_batch = [masks_g[id] for id in batch]

            train_batch_x = np.array(profile_batch).reshape(size, len(all_movies) * 5)
            train_batch_o = np.array(occ_batch).reshape(size, len(all_occupations))
            train_batch_s = np.array(sex_batch).reshape(size, 1)
            train_batch_a = np.array(age_batch).reshape(size, len(all_ages))
            train_batch_g = np.array(genre_batch).reshape(size, len(all_movies) * len(all_genres))
            train_masks_x = np.array(masks_x_batch).reshape(size, len(all_movies) * 5)
            train_masks_o = np.array(masks_o_batch).reshape(size, len(all_occupations))
            train_masks_s = np.array(masks_s_batch).reshape(size, 1)
            train_masks_a = np.array(masks_a_batch).reshape(size, len(all_ages))
            train_masks_g = np.array(masks_g_batch).reshape(size, len(all_movies) * len(all_genres))
            train_masks_x = train_masks_x.astype('float32')
            train_masks_o = train_masks_o.astype('float32')
            train_masks_s = train_masks_s.astype('float32')
            train_masks_a = train_masks_a.astype('float32')
            train_masks_g = train_masks_g.astype('float32')
            train(train_batch_x, train_batch_o, train_batch_s, train_batch_a, train_batch_g, train_masks_x, train_masks_o, train_masks_s, train_masks_a, train_masks_g)
            sys.stdout.write('.')
            sys.stdout.flush()

        end_time = time.time()
        train_time = end_time - start_time

        ratings = []
        predictions = []

        start_time = time.time()

        for batch in chunker(tests.keys(), batch_size):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            occ_profiles = {}
            sex_profiles = {}
            age_profiles = {}
            genre_profiles = {}

            for userid in batch:
                user_profile = [0.] * len(all_movies)
                occ_profile = [0.] * len(all_occupations)
                sex_profile = [0.] * 1
                age_profile = [0.] * len(all_ages)
                genre_profile = [0.] * len(all_movies) * len(all_genres)

                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(len(all_genres)):
                        genre_profile[len(all_genres) * all_movies.index(movie_id) + _i] = movie_genres[len(all_genres) * all_movies.index(movie_id) + _i]

                example_x = expand(np.array([user_profile])).astype('float32')
                example_o = expand(np.array([occ_profile]), k=1).astype('float32')
                example_s = expand(np.array([sex_profile]), k=1).astype('float32')
                example_a = expand(np.array([age_profile]), k=1).astype('float32')
                example_g = np.array(genre_profile).astype('float32')

                bin_profiles[userid] = example_x
                occ_profiles[userid] = example_o
                sex_profiles[userid] = example_s
                age_profiles[userid] = example_a
                genre_profiles[userid] = example_g

            positions = {profile_id: pos for pos, profile_id
                         in enumerate(batch)}
            profile_batch = [bin_profiles[el] for el in batch]
            occ_batch = [occ_profiles[el] for el in batch]
            sex_batch = [sex_profiles[el] for el in batch]
            age_batch = [age_profiles[el] for el in batch]
            genre_batch = [genre_profiles[id] for id in batch]
            test_batch_x = np.array(profile_batch).reshape(size, len(all_movies) * 5)
            test_batch_o = np.array(occ_batch).reshape(size, len(all_occupations))
            test_batch_s = np.array(sex_batch).reshape(size, 1)
            test_batch_a = np.array(age_batch).reshape(size, len(all_ages))
            test_batch_g = np.array(genre_batch).reshape(size, len(all_movies) * len(all_genres))
            user_preds = revert_expected_value(predict(test_batch_x, test_batch_o, test_batch_s, test_batch_a, test_batch_g))
            for profile_id in batch:
                test_movies = tests[profile_id]
                try:
                    for movie, rating in test_movies:
                        current_profile = user_preds[positions[profile_id]]
                        predicted = current_profile[all_movies.index(movie)]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        end_time = time.time()
        test_time = end_time - start_time

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        mae = vabs(distances).mean()
        rmse = sqrt((distances ** 2).mean())

        iteration_result = {
            'iteration': j,
            'k': k,
            'momentum': momentum,
            'mae': mae,
            'rmse': rmse,
            'lrate': current_l_w,
            'train_time': train_time,
            'test_time': test_time
        }

        config_result['results'].append(iteration_result)

        print(iteration_str.format(j, k, current_l_w, momentum, mae, rmse))

        with open('experiments/{}_{}.json'.format(config_name, name), 'wt') as res_output:
            res_output.write(json.dumps(config_result, indent=4))

if __name__ == "__main__":
    experiments = read_experiment(sys.argv[1])

    for experiment in experiments:
        name = experiment['name']
        train_path = experiment['train_path']
        user_path = experiment['user_path']
        item_path = experiment['item_path']
        test_path = experiment['test_path']
        sep = experiment['sep']
        configs = experiment['configs']

        all_users, all_movies, all_occupations, all_sex, all_ages, all_genres, tests = load_data(train_path, test_path, user_path, item_path, sep, user_based=True)

        for config in configs:
            run(name, train_path, user_path, item_path, config, all_users, all_movies, all_occupations, all_sex, all_ages, all_genres, tests, None, sep)
