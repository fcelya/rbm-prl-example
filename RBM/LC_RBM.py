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

import numpy as np

def read_experiment(path):

    with open(path) as data:
        experiments = json.loads(data.read())['experiments']

    return experiments



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

    return np.array(users).reshape(m.shape[0], m.shape[1] // k)


def load_dataset(train_path, test_path, item_path, sep, user_based=True):
    all_users_set = set()
    all_movies_set = set()
    all_genres_set = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

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


    return list(all_users_set), list(all_movies_set), list(all_genres_set), tests


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



x = tt.matrix()
y = tt.matrix()


def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]


def cast32(x):
    return tt.cast(x, 'float32')


class CFRBM:
    def __init__(self, num_visible_x, num_visible_g, num_hidden, initial_v_x=None, initial_v_g=None, initial_weigths_x=None, initial_weights_g=None, debug=False):
        self.dim_x = (num_visible_x, num_hidden)
        self.dim_g = (num_visible_g, num_hidden)
        self.num_visible_x = num_visible_x
        self.num_visible_g = num_visible_g
        self.num_hidden = num_hidden

        if initial_weigths_x:
            initial_weights_x = np.load('{}.W_x.npy'.format(initial_weigths_x))
            initial_weigths_g = np.load('{}.W_g.npy'.format(initial_weigths_g))
            initial_hbias = np.load('{}.h.npy'.format(initial_weigths_x))
            initial_vbias_x = np.load('{}.b_x.npy'.format(initial_weigths_x))
            initial_vbias_g = np.load('{}.b_g.npy'.format(initial_weigths_g))
        else:
            initial_weights_x = np.array(np.random.normal(0, 0.1, size=self.dim_x), dtype=np.float32)
            initial_weights_g = np.array(np.random.normal(0, 0.1, size=self.dim_g), dtype=np.float32)
            initial_hbias = np.zeros(num_hidden, dtype=np.float32)

            if initial_v_x:
                initial_vbias_x = np.array(initial_v_x, dtype=np.float32)
                initial_vbias_g = np.array(initial_v_g, dtype=np.float32)
            else:
                initial_vbias_x = np.zeros(num_visible_x, dtype=np.float32)
                initial_vbias_g = np.zeros(num_visible_g, dtype=np.float32)

        self.weights_x = theano.shared(value=initial_weights_x, borrow=True, name='weights_x')
        self.weights_g = theano.shared(value=initial_weights_g, borrow=True, name='weights_g')
        self.vbias_x = theano.shared(value=initial_vbias_x, borrow=True, name='vbias_x')
        self.vbias_g = theano.shared(value=initial_vbias_g, borrow=True, name='vbias_g')
        self.hbias = theano.shared(value=initial_hbias, borrow=True, name='hbias')

        prev_gw_x = np.zeros(shape=self.dim_x, dtype=np.float32)
        prev_gh = np.zeros(num_hidden, dtype=np.float32)
        prev_gw_g = np.zeros(shape=self.dim_g, dtype=np.float32)
        prev_gv_x = np.zeros(num_visible_x, dtype=np.float32)
        prev_gv_g = np.zeros(num_visible_g, dtype=np.float32)
        
        self.prev_gw_g = theano.shared(value=prev_gw_g, borrow=True, name='g_w_g')
        self.prev_gw_x = theano.shared(value=prev_gw_x, borrow=True, name='g_w_x')
        self.prev_gh = theano.shared(value=prev_gh, borrow=True, name='g_h')
        self.prev_gv_x = theano.shared(value=prev_gv_x, borrow=True, name='g_v_x')
        self.prev_gv_g = theano.shared(value=prev_gv_g, borrow=True, name='g_v_g')

        self.theano_rng = tt.shared_randomstreams.RandomStreams(np.random.RandomState(17).randint(2**30))

        if debug:
            theano.config.compute_test_value = 'warn'
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prop_up(self, vis, gen):
        return tt.nnet.sigmoid(tt.dot(vis, self.weights_x) + tt.dot(gen, self.weights_g) + self.hbias)

    def sample_hidden(self, vis, gen):
        activations = self.prop_up(vis, gen)
        h1_sample = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)
        return h1_sample, activations

    def prop_down_x(self, h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_x.T) + self.vbias_x)

    def prop_down_g(self, h):
        return tt.nnet.sigmoid(tt.dot(h, self.weights_g.T) + self.vbias_g)

    def sample_visible_x(self, h, k=5):
        activations = self.prop_down_x(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_x = activations / partitions.reshape(activations.shape)
        v1_sample_x = self.theano_rng.binomial(size=activations.shape, n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample_x, activations_x

    def sample_visible_g(self, h, k=1):
        activations = self.prop_down_g(h)
        k_ones = tt.ones(k)

        partitions = activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_g = activations / partitions.reshape(activations.shape)
        v1_sample_g = self.theano_rng.binomial(size=activations.shape,  n=1, p=activations, dtype=theano.config.floatX)

        return v1_sample_g, activations_g

    def contrastive_divergence_1(self, v1_x, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_g)
        v2_x, v2a_x = self.sample_visible_x(h1)
        v2_g, v2a_g = self.sample_visible_g(h1)
        h2, h2a = self.sample_hidden(v2_x, v2_g)

        return (v1_x, v1_g, h1, v2_x, v2a_x, v2_g, v2a_g, h2, h2a)

    def gradient(self, v1_x, v1_g, h1, v2_x, v2_g, h2p, masks_x, masks_g):
        v1_xh1_mask_x = outer(masks_x, h1)
        v1_gh1_mask_g = outer(masks_g, h1)

        gw_x = ((outer(v1_x, h1) * v1_xh1_mask_x) - (outer(v2_x, h2p) * v1_xh1_mask_x)).mean(axis=0)
        gw_g = ((outer(v1_g, h1) * v1_gh1_mask_g) - (outer(v2_g, h2p) * v1_gh1_mask_g)).mean(axis=0)
        gv_x = ((v1_x * masks_x) - (v2_x * masks_x)).mean(axis=0)
        gv_g = ((v1_g * masks_g) - (v2_g * masks_g)).mean(axis=0)
        gh = (h1 - h2p).mean(axis=0)

        return (gw_x, gw_g, gv_x, gv_g, gh)

    def cdk_fun(self, vis_x, vis_g, masks_x, masks_g, k=1, w_lr=0.000021, v_lr=0.000025, h_lr=0.000025, decay=0.0000, momentum=0.0):
        v1_x, v1_g, h1, v2_x, v2a_x, v2_g, v2a_g, h2, h2a = self.contrastive_divergence_1(vis_x, vis_g)

        for i in range(k-1):
            v1_x, v1_g, h1, v2_x, v2a_x, v2_g, v2a_g, h2, h2a = self.contrastive_divergence_1(v2_x, v2_g)

        (W_x, W_g, V_x, V_g, H) = self.gradient(v1_x, v1_g, h1, v2_x, v2_g, h2a, masks_x, masks_g)

        if decay:
            W_x -= decay * self.weights_x
            W_g -= decay * self.weights_g

        updates = [
            (self.weights_x, cast32(self.weights_x + (momentum * self.prev_gw_x) + (W_x * w_lr))),
            (self.weights_g, cast32(self.weights_g + (momentum * self.prev_gw_g) + (W_g * w_lr))),
            (self.vbias_x, cast32(self.vbias_x + (momentum * self.prev_gv_x) + (V_x * v_lr))),
            (self.vbias_g, cast32(self.vbias_g + (momentum * self.prev_gv_g) + (V_g * v_lr))),
            (self.hbias, cast32(self.hbias + (momentum * self.prev_gh) + (H * h_lr))),
            (self.prev_gw_x, cast32(W_x)),
            (self.prev_gw_g, cast32(W_g)),
            (self.prev_gh, cast32(H)),
            (self.prev_gv_x, cast32(V_x)),
            (self.prev_gv_g, cast32(V_g))
        ]

        return theano.function([vis_x, vis_g, masks_x, masks_g], updates=updates)

    def predict(self, v1_x, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_g)
        v2_x, v2a_x = self.sample_visible_x(h1)
        return theano.function([v1_x, v1_g], v2a_x)


def run(name, dataset, movie_info, config, all_users, all_movies, all_genres, tests, initial_v, sep):
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
    vis_g = tt.matrix()
    vmasks_x = tt.matrix()
    vmasks_g = tt.matrix()

    rbm = CFRBM(len(all_users) * 5, len(all_genres), number_hidden)

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[mid].append((uid, float(rat)))

    print("Users and ratings loaded")

    movie_genres = defaultdict(list)
    
    r = csv.reader(open(movie_info, 'rt',encoding = "ISO-8859-1"), delimiter='|')
    for row in r:
        movie_genres[row[0]] = [int(x) for x in row[5:]]

    print("Movie genres loaded")

    for j in range(epochs):
        def get_index(col):
            if j/(epochs/len(col)) < len(col):
                return j/(epochs/len(col))
            else:
                return -1

        index  = int(get_index(ks))
        mindex = int(get_index(momentums))
        icurrent_l_w = int(get_index(l_w))
        icurrent_l_v = int(get_index(l_v))
        icurrent_l_h = int(get_index(l_h))

        k = ks[index]
        momentum = momentums[mindex]
        current_l_w = l_w[icurrent_l_w]
        current_l_v = l_v[icurrent_l_v]
        current_l_h = l_h[icurrent_l_h]

        train = rbm.cdk_fun(vis_x, vis_g, vmasks_x, vmasks_g, k=k, w_lr=current_l_w, v_lr=current_l_v, h_lr=current_l_h, decay=decay, momentum=momentum)
        predict = rbm.predict(vis_x,vis_g)

        start_time = time.time()
        for batch_i, batch in enumerate(chunker(list(profiles.keys()), batch_size)):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            gen_profiles = {}
            masks_x = {}
            masks_g = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                genre_profile = [0.] * len(all_genres)
                mask_x = [0] * (len(all_users) * 5)
                mask_g = [0] * (len(all_genres))

                for user_id, rat in profiles[movieid]:
                    movie_profile[all_users.index(user_id)] = rat
                    for _i in range(5):
                        mask_x[5 * all_users.index(user_id) + _i] = 1
                
                mask_g = [1] * len(all_genres)

                example_x = expand(np.array([movie_profile])).astype('float32')
                example_g = expand(np.array([genre_profile]), k=1).astype('float32')
                bin_profiles[movieid] = example_x
                gen_profiles[movieid] = example_g
                masks_x[movieid] = mask_x
                masks_g[movieid] = mask_g

            movies_batch = [bin_profiles[id] for id in batch]
            genres_batch = [gen_profiles[id] for id in batch]
            masks_x_batch = [masks_x[id] for id in batch]
            masks_g_batch = [masks_g[id] for id in batch]

            train_batch_x = np.array(movies_batch).reshape(size, len(all_users) * 5)
            train_batch_g = np.array(genres_batch).reshape(size, len(all_genres))
            train_masks_x = np.array(masks_x_batch).reshape(size, len(all_users) * 5)
            train_masks_g = np.array(masks_g_batch).reshape(size, len(all_genres))
            train_masks_x = train_masks_x.astype('float32')
            train_masks_g = train_masks_g.astype('float32')
            train(train_batch_x, train_batch_g, train_masks_x, train_masks_g)
            sys.stdout.write('.')
            sys.stdout.flush()

        end_time = time.time()
        train_time = end_time - start_time

        ratings = []
        predictions = []

        start_time = time.time()

        for batch in chunker(list(tests.keys()), batch_size):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            gen_profiles = {}
            masks_x = {}
            masks_g = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                genre_profile = [0.] * len(all_genres)
                mask_x = [0] * (len(all_users) * 5)
                mask_g = [0] * (len(all_genres))

                for userid, rat in profiles[movieid]:
                    movie_profile[all_users.index(userid)] = rat
                    for _i in range(5):
                        mask_x[5 * all_users.index(userid) + _i] = 1

                mask_g = [1] * len(all_genres)

                example_x = expand(np.array([movie_profile])).astype('float32')
                example_g = expand(np.array([genre_profile]), k=1).astype('float32')
                bin_profiles[movieid] = example_x
                gen_profiles[movieid] = example_g
                masks_x[movieid] = mask_x
                masks_g[movieid] = mask_g


            positions = {movie_id: pos for pos, movie_id in enumerate(batch)}
            movies_batch = [bin_profiles[el] for el in batch]
            genres_batch = [gen_profiles[el] for el in batch]
            test_batch_x = np.array(movies_batch).reshape(size, len(all_users) * 5)
            test_batch_g = np.array(genres_batch).reshape(size, len(all_genres))
            movie_predictions = revert_expected_value(predict(test_batch_x, test_batch_g))

            for movie_id in batch:
                test_users = tests[movie_id]
                try:
                    for user, rating in test_users:
                        current_movie = movie_predictions[positions[movie_id]]
                        predicted = current_movie[all_users.index(user)]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        end_time = time.time()
        test_time = end_time - start_time

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        true_rat = np.array(ratings, dtype=np.uint8)
        pred_rat = np.array(predictions, dtype=np.uint8)

        #print true_rat < 3, true_rat
        prec_rec = precision_recall_fscore_support(true_rat < 3,pred_rat < 3, average='binary')
        print(prec_rec)

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
            'test_time': test_time,
            'prec_rec': prec_rec
        }

        config_result['results'].append(iteration_result)

        print(iteration_str.format(j, k, current_l_w, momentum, mae, rmse))

        with open('{}_{}.json'.format(config_name, name), 'wt') as res_output:
            res_output.write(json.dumps(config_result, indent=4))

if __name__ == "__main__":

    experiments = read_experiment(sys.argv[1])

    for experiment in experiments:
        name = experiment['name']
        train_path = experiment['train_path']
        item_path = experiment['item_path']
        test_path = experiment['test_path']
        sep = experiment['sep']

        all_users, all_movies, all_genres,tests = load_dataset(train_path, test_path, item_path, sep, user_based=False)

        for config in experiment['configs']:
            run(name, train_path, item_path, config, all_users, all_movies, all_genres, tests, None, sep)

