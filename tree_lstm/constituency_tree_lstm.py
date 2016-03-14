from optparse import OptionParser

import os
import sys
import json
import codecs
import gensim
import random
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from read_data import load_ptb_trees

# Otherwise the deepcopy fails
sys.setrecursionlimit(5000)

THEANO_FLAGS='floatX=float32'

class ConstituencyTreeLSTM(object):

    def __init__(self, dv, dh, dx, nc, alpha=1.0, init_scale=0.2, initial_embeddings=None, params_init=None,
                 update='adagrad', seed=None, drop_p=0.5, momentum=0.9):

        self.dv = dv  # vocabulary size
        self.dh = dh  # hidden node size
        self.dx = dx  # word embedding size
        self.nc = nc  # number of classes
        self.alpha = alpha  # regularization strength
        self.drop_p = drop_p  # probability of dropping an input with dropout

        # adagrad parameters
        self.epsilon = 0.00001

        if initial_embeddings is None:
            self.emb = theano.shared(name='embeddings',
                                     value=init_scale * np.random.uniform(-1.0, 1.0,
                                                                          (dv, dx)).astype(theano.config.floatX))
        else:
            self.emb = theano.shared(name='embeddings', value=initial_embeddings.astype(theano.config.floatX))

        self.W_x_i = theano.shared(name='W_x_i', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, dh))
                                   .astype(theano.config.floatX))
        self.W_hl_i = theano.shared(name='W_hl_i', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.W_hr_i = theano.shared(name='W_hr_i', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.b_h_i = theano.shared(name='b_h_i', value=np.array(np.zeros(dh),
                                                                dtype=theano.config.floatX))

        self.W_x_f = theano.shared(name='W_x_f', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, dh))
                                   .astype(theano.config.floatX))
        self.b_h_f = theano.shared(name='b_h_f', value=np.array(np.random.uniform(0.0, 1.0, dh),
                                                                dtype=theano.config.floatX))

        self.W_hl_fl = theano.shared(name='W_hl_fl', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                     .astype(theano.config.floatX))
        self.W_hr_fl = theano.shared(name='W_hr_fl', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                     .astype(theano.config.floatX))

        self.W_hl_fr = theano.shared(name='W_hl_fr', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                     .astype(theano.config.floatX))
        self.W_hr_fr = theano.shared(name='W_hr_fr', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                     .astype(theano.config.floatX))

        self.W_x_o = theano.shared(name='W_x_o', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, dh))
                                   .astype(theano.config.floatX))
        self.W_hl_o = theano.shared(name='W_hl_o', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.W_hr_o = theano.shared(name='W_hr_o', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.b_h_o = theano.shared(name='b_h_o', value=np.array(np.zeros(dh),
                                                                dtype=theano.config.floatX))

        self.W_x_u = theano.shared(name='W_x_u', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, dh))
                                   .astype(theano.config.floatX))
        self.W_hl_u = theano.shared(name='W_hl_u', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.W_hr_u = theano.shared(name='W_hr_u', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, dh))
                                    .astype(theano.config.floatX))
        self.b_h_u = theano.shared(name='b_h_u', value=np.array(np.zeros(dh),
                                                                dtype=theano.config.floatX))

        self.W_z = theano.shared(name='W_z', value=init_scale * np.random.uniform(-1.0, 1.0, (dh, nc))
                                 .astype(theano.config.floatX))
        self.b_z = theano.shared(name='b_z', value=np.array(np.zeros(nc),
                                                            dtype=theano.config.floatX))

        self.params = [self.W_x_i, self.W_hl_i, self.W_hr_i, self.b_h_i]
        self.params += [self.W_x_f, self.W_hl_fl, self.W_hr_fl, self.b_h_f]
        self.params += [self.W_hl_fr, self.W_hr_fr]
        self.params += [self.W_x_o, self.W_hl_o, self.W_hr_o, self.b_h_o]
        self.params += [self.W_x_u, self.W_hl_u, self.W_hr_u, self.b_h_u]
        self.params += [self.W_z, self.b_z]

        self.param_shapes = [(dx, dh), (dh, dh), (dh, dh), (dh,),
                             (dx, dh), (dh, dh), (dh, dh), (dh,),
                             (dh, dh), (dh, dh),
                             (dx, dh), (dh, dh), (dh, dh), (dh,),
                             (dx, dh), (dh, dh), (dh, dh), (dh,),
                             (dh, nc), (nc,)]

        if update == 'adagrad':
            self.grad_histories = [
                theano.shared(
                    value=np.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="grad_hist:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.params)
                ]

        elif update == 'sgdm':
            self.velocity = [
                theano.shared(
                    value=np.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="momentum:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.params)
                ]
            self.momentum = momentum

        self.theano_rng = RandomStreams(seed)

        idxs = T.ivector()
        sequence_length = T.shape(idxs)[0]
        temp = self.emb[idxs]
        x = temp.reshape([sequence_length, dx])

        #counter = T.ivector('counter')
        left_mask = T.imatrix('left_mask')
        right_mask = T.imatrix('right_mask')
        y = T.iscalar('y')
        lr = T.scalar('lr', dtype=theano.config.floatX)
        is_train = T.iscalar('is_train')
        drop_x = T.iscalar('drop_x')

        # This is a bit annoying; the 0th dimension of x needs to be sequence, so we can iterate over it
        # but the 0th dimension of the hidden nodes needs to be hidden-node dimension, so that we can broadcast
        # the mask out to it
        def treefwd(x_t, left_mask_t, right_mask_t, counter_t, h_tm1, c_tm1):
            h_t = h_tm1
            c_t = c_tm1
            # zero out the input unless this is a leaf node
            input = T.switch(T.eq(T.sum(left_mask_t) + T.sum(right_mask_t), 0), x_t, x_t*0)
            i_t = T.nnet.sigmoid(T.dot(input, self.W_x_i) + T.sum(T.dot(self.W_hl_i.T, (left_mask_t * h_tm1)).T, axis=0) + T.sum(T.dot(self.W_hr_i.T, (right_mask_t * h_tm1)).T, axis=0) + self.b_h_i)
            fl_t = T.nnet.sigmoid(T.dot(input, self.W_x_f) + T.sum(T.dot(self.W_hl_fl.T, (left_mask_t * h_tm1)).T, axis=0) + T.sum(T.dot(self.W_hr_fl.T, (right_mask_t * h_tm1)).T, axis=0) + self.b_h_f)
            fr_t = T.nnet.sigmoid(T.dot(input, self.W_x_f) + T.sum(T.dot(self.W_hl_fr.T, (left_mask_t * h_tm1)).T, axis=0) + T.sum(T.dot(self.W_hr_fr.T, (right_mask_t * h_tm1)).T, axis=0) + self.b_h_f)
            o_t = T.nnet.sigmoid(T.dot(input, self.W_x_o) + T.sum(T.dot(self.W_hl_o.T, (left_mask_t * h_tm1)).T, axis=0) + T.sum(T.dot(self.W_hr_o.T, (right_mask_t * h_tm1)).T, axis=0) + self.b_h_o)
            u_t = T.tanh(T.dot(input, self.W_x_u) + T.sum(T.dot(self.W_hl_u.T, (left_mask_t * h_tm1)).T, axis=0) + T.sum(T.dot(self.W_hr_u.T, (right_mask_t * h_tm1)).T, axis=0) + self.b_h_u)
            c_temp = i_t * u_t + fl_t * T.sum((left_mask_t * c_tm1).T, axis=0) + fr_t * T.sum((right_mask_t * c_tm1).T, axis=0)
            h_temp = o_t * T.tanh(c_temp)
            h_t = T.set_subtensor(h_t[:, counter_t], h_temp)
            c_t = T.set_subtensor(c_t[:, counter_t], c_temp)
            return h_t, c_t

        def drop(drop_input, drop_p, is_train):
            mask = self.theano_rng.binomial(p=1.0-drop_p, size=drop_input.shape, dtype=theano.config.floatX)
            return T.cast(T.switch(T.neq(is_train, 0), drop_input * mask, drop_input * (1.0-self.drop_p)), dtype=theano.config.floatX)

        ds, dx = T.shape(x)
        # do dropout on x, if specified
        x = T.switch(T.neq(drop_x, 0), drop(x, self.drop_p, is_train), x)
        output, _ = theano.scan(fn=treefwd, sequences=[x, left_mask, right_mask, T.arange(0, ds)], outputs_info=[T.zeros((dh, ds), dtype=theano.config.floatX), T.zeros((dh, ds), dtype=theano.config.floatX)])
        full_h, full_c = output
        h = full_h[-1, :, -1]

        h = drop(h, self.drop_p, is_train)
        temp = T.dot(h, self.W_z) + self.b_z
        p_y_given_x = T.nnet.softmax(temp)[0]
        pred_y = T.argmax(p_y_given_x)

        log_loss = T.sum(-T.log(p_y_given_x[y]))
        penalty = T.sum([T.sum(p ** 2) for p in self.params])
        cost = log_loss + alpha * penalty / 2.0

        gradients = [T.grad(cost, param) for param in self.params]

        if update == 'adagrad':
            new_grad_histories = [
                T.cast(g_hist + g ** 2, dtype=theano.config.floatX)
                for g_hist, g in zip(self.grad_histories, gradients)
                ]
            grad_hist_update = zip(self.grad_histories, new_grad_histories)

            param_updates = [(param, T.cast(param - lr / (T.sqrt(g_hist) + self.epsilon) * param_grad, dtype=theano.config.floatX))
                             for param, param_grad, g_hist in zip(self.params, gradients, new_grad_histories)]

            updates = grad_hist_update + param_updates

        # sgd with momentum
        elif update == 'sgdm':
            velocity_t = [momentum * v + lr * g for v, g in zip(self.velocity, gradients)]
            velocity_updates = [(v, T.cast(v_t, theano.config.floatX)) for v, v_t in zip(self.velocity, velocity_t)]
            param_updates = [(param, T.cast(param - v_t, theano.config.floatX)) for param, v_t in zip(self.params, velocity_t)]
            updates = velocity_updates + param_updates

        # else, basic sgd
        else:
            updates = OrderedDict((p, T.cast(p - lr * g, dtype=theano.config.floatX)) for p, g in zip(self.params, gradients))

        self.train = theano.function(inputs=[idxs, left_mask, right_mask, y, lr, is_train, drop_x],
                                     outputs=[pred_y, p_y_given_x, log_loss, cost], updates=updates,
                                     on_unused_input='ignore')
        self.predict = theano.function(inputs=[idxs, left_mask, right_mask, is_train, drop_x],
                                       outputs=[pred_y, p_y_given_x])

        # good example of how to see a value in a tensor; way easier than theano.printing.Print()
        idx = T.iscalar('idx')
        emb = self.emb[idx]
        self.get_embedding = theano.function(inputs=[idx], outputs=emb)

    def print_params(self):
        for param in self.params:
            print param.name, param.get_value()


def main():
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', dest='alpha', default=0.000004,
                      help='Regularization strength: default=%default')
    parser.add_option('-d', dest='hidden_dim', default=150,
                      help='Hidden node dimension: default=%default')
    parser.add_option('-e', dest='epochs', default=20,
                      help='Number of epochs: default=%default')
    parser.add_option('-i', dest='iter_display', default=500,
                      help='Number of iterations between output: default=%default')
    parser.add_option('-o', dest='optimization', default='adagrad',
                      help='Optimization method [sgd|sgdm|adagrad]: default=%default')
    parser.add_option('-l', dest='learning_rate', default=0.002,
                      help='Initial learning rate: default=%default')
    parser.add_option('--decay', dest='decay', default=0.95,
                      help='Learning rate decay (sgd|sgdm only): default=%default')
    parser.add_option('--momentum', dest='momentum', default=0.5,
                      help='Momentum parameter (sgdm only): default=%default')
    parser.add_option('-s', dest='seed', default=42,
                      help='Random seed: default=%default')
    parser.add_option('--word2vec_file', dest='word2vec_file', default='',
                      help='Location of word2vec file: default=reload previously saved subset of vectors')
    parser.add_option('--save_vectors', action="store_true", dest="save_vectors", default=False,
                      help='Save relevant word vectors (for faster loading next time)')
    parser.add_option('--train_root_only', action="store_true", dest="train_root_only", default=False,
                      help='Train only on the full sentences (not subtrees): default=%default')
    parser.add_option('--no_eval', action="store_true", dest="no_eval", default=False,
                      help='Skip the evaluation between epochs: default=%default')
    parser.add_option('--drop_x', action="store_true", dest="drop_x", default=False,
                      help='Add dropout to the input layer: default=%default')
    parser.add_option('--five_class', action="store_true", dest="five_class", default=False,
                      help='Do five class classification (default is binary)')

    (options, args) = parser.parse_args()
    input_dir = args[0]

    seed = int(options.seed)
    n_epochs = int(options.epochs)
    alpha = float(options.alpha)
    lr = float(options.learning_rate)
    iter_display = int(options.iter_display)
    opti_method = options.optimization
    lr_decay = float(options.decay)
    momentum = float(options.momentum)
    word2vec_file = options.word2vec_file
    save_vectors = options.save_vectors
    train_root_only = options.train_root_only
    no_eval = options.no_eval
    drop_x = int(options.drop_x)
    five_class = options.five_class

    if five_class:
        binary = False
        nc = 5
    else:
        binary = True
        nc = 2

    if seed > 0:
        np.random.seed(seed)
        random.seed(seed)

    dh = int(options.hidden_dim)
    dx = 300

    # load sentiment trees
    print "Loading trees"
    if train_root_only:
        trees, train_vocab = load_ptb_trees(input_dir, "train", binary=binary, root_only=True)
    else:
        trees, train_vocab = load_ptb_trees(input_dir, "train", binary=binary, root_only=False)
    print len(trees), "train trees loaded"

    train_root_trees, _ = load_ptb_trees(input_dir, "train", binary=binary, root_only=True)
    print len(train_root_trees), "train root trees loaded"

    dev_root_trees, dev_vocab = load_ptb_trees(input_dir, "dev", binary=binary, root_only=True)
    print len(dev_root_trees), "dev root trees loaded"

    test_root_trees, test_vocab = load_ptb_trees(input_dir, "test", binary=binary, root_only=True)
    print len(test_root_trees), "test root trees loaded"

    if word2vec_file != '':
        # load pre-trained word vectors
        print "Loading word2vec word vectors"
        vectors = gensim.models.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

        pruned_vocab = set()
        for v in list(train_vocab):
            if v in vectors:
                pruned_vocab.add(v)
        pruned_vocab.add('_')
        pruned_vocab.add('_OOV_')
        vocab = list(pruned_vocab)
        vocab.sort()
        vocab_size = len(vocab)
        vocab_index = dict(zip(vocab, range(vocab_size)))

        print "Preparing word vectors"
        total_count = 0
        emb_dim = dx
        initial_embeddings = np.zeros([vocab_size, emb_dim], dtype=float)

        for v, i in vocab_index.items():
            total_count += 1
            if v == '_':
                initial_embeddings[i, :] = np.zeros(emb_dim)
            elif v == '_OOV_':
                initial_embeddings[i, :] = 0.05 * np.random.uniform(-1.0, 1.0, (1, emb_dim))
            elif v in vectors:
                initial_embeddings[i, :] = vectors[v]
            else:
                sys.exit('word not in vocab')

        if save_vectors:
            print "Saving word vectors"
            pickle_data(initial_embeddings, os.path.join(input_dir, 'initial_embeddings.pkl'))
            write_to_json(vocab, os.path.join(input_dir, 'vocab.json'))
    else:
        print "Loading relevant word vectors"
        initial_embeddings = unpickle_data(os.path.join(input_dir, 'initial_embeddings.pkl'))
        vocab = read_json(os.path.join(input_dir, 'vocab.json'))
        vocab.sort()
        vocab_size = len(vocab)
        vocab_index = dict(zip(vocab, range(vocab_size)))

    print len(vocab), "words in pruned vocab"
    print len(list(train_vocab - set(vocab))), "words missing from word vectors in training vocabulary"
    print len(list(dev_vocab - set(vocab))), "words missing from word vectors in dev vocabulary"
    print len(list(test_vocab - set(vocab))), "words missing from word vectors in test vocabulary"

    print "Indexing words"
    for k, t in trees.items():
        t['idxs'] = [vocab_index[w] if w in vocab_index else vocab_index['_OOV_'] for w in t['words']]
    for k, t in dev_root_trees.items():
        t['idxs'] = [vocab_index[w] if w in vocab_index else vocab_index['_OOV_'] for w in t['words']]
    for k, t in train_root_trees.items():
        t['idxs'] = [vocab_index[w] if w in vocab_index else vocab_index['_OOV_'] for w in t['words']]

    # create the LSTM
    theano_seed = np.random.randint(2 ** 30)
    ctreeLSTM = ConstituencyTreeLSTM(vocab_size, dh, dx, nc, initial_embeddings=initial_embeddings, alpha=alpha,
                                     update=opti_method, seed=theano_seed, momentum=momentum)
    keys = trees.keys()
    random.shuffle(keys)

    if not no_eval:
        print "Pre-training evaluation"
        train_z_o_loss, train_log_loss = evaluate(train_root_trees, ctreeLSTM, drop_x)
        valid_z_o_loss, valid_log_loss = evaluate(dev_root_trees, ctreeLSTM, drop_x)
        test_z_o_loss, test_log_loss = evaluate(test_root_trees, ctreeLSTM, drop_x)
        print ('epoch=%d\ttrain_0/1=%.3f\ttrain_log=%.3f\tdev_0/1=%.3f\tdev_log=%.3f\ttest_0/1=%.3f\ttest_log=%.3f') % \
              (-1, train_z_o_loss, train_log_loss, valid_z_o_loss, valid_log_loss, test_z_o_loss, test_log_loss)

    print "Training"
    for epoch in range(n_epochs):
        sum_log_loss = 0
        sum_loss = 0
        mistakes = 0
        pred0 = 0
        pred1 = 0
        print "epoch\titems\tloss\tl+reg\terrs\tpredict 1"
        for k_i, t_i in enumerate(keys):
            t = trees[t_i]
            idxs = t['idxs']
            left_mask = t['left_mask']
            right_mask = t['right_mask']
            value = t['value']

            counter = np.array(np.arange(0, len(idxs)), dtype=np.int32)
            pred_y, p_y_given_x, log_loss, loss = ctreeLSTM.train(idxs, left_mask, right_mask, value, lr, 1, drop_x)
            sum_log_loss += log_loss
            sum_loss += loss
            pred0 += value
            pred1 += (1-value)
            if pred_y != value:
                mistakes += 1
            if k_i % iter_display == 0:
                d = float(k_i+1)
                #print t['words'], value, pred_y, p_y_given_x
                print '%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f' % \
                      (epoch, k_i, sum_log_loss/d, sum_loss/d, mistakes/d, pred1/d)

        if not no_eval:
            train_z_o_loss, train_log_loss = evaluate(train_root_trees, ctreeLSTM, drop_x)
            valid_z_o_loss, valid_log_loss = evaluate(dev_root_trees, ctreeLSTM, drop_x)
            test_z_o_loss, test_log_loss = evaluate(test_root_trees, ctreeLSTM, drop_x)
            print ('epoch=%d\ttrain_0/1=%.3f\ttrain_log=%.3f\tdev_0/1=%.3f\tdev_log=%.3f\ttest_0/1=%.3f\ttest_log=%.3f') % \
                  (epoch, train_z_o_loss, train_log_loss, valid_z_o_loss, valid_log_loss, test_z_o_loss, test_log_loss)

        if opti_method != 'adagrad':
            lr *= lr_decay


def evaluate(trees, rnn, drop_x):
    zero_one_loss = 0
    log_loss = 0
    n_trees = len(trees)
    for k_i, key in enumerate(trees.keys()):
        t = trees[key]
        idxs = t['idxs']
        left_mask = t['left_mask']
        right_mask = t['right_mask']
        true_y = t['value']

        pred_y, p_y_given_x = rnn.predict(idxs, left_mask, right_mask, 0, drop_x)

        log_loss += -np.log(p_y_given_x[true_y])
        if pred_y != true_y:
            zero_one_loss += 1

        if k_i % 1000 == 0 and k_i > 0:
            print k_i, log_loss, zero_one_loss

    return zero_one_loss/float(n_trees), log_loss/float(n_trees)


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        temp = json.load(input_file, encoding='utf-8')
        if isinstance(temp, dict):
            # try convert keys back to ints, if appropriate
            try:
                data = {int(k): v for (k, v) in temp.items()}
            except ValueError, e:
                data = temp
        else:
            data = temp
    return data


if __name__ == '__main__':
    main()
