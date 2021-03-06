from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, BNF
from gcn.Data_processing import load_data

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../Data/Autism/', 'Dataset string.')
flags.DEFINE_string('idcsv', '../Data/Autism/labels.csv', 'Dataset id.')
flags.DEFINE_string('model', 'BNF', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('nodesize', 246, 'Node size of Adj matrix')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.4, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data

train_features, adj, train_labels, test_features, test_labels = load_data(FLAGS.dataset,FLAGS.idcsv)


# Some preprocessing
# features = preprocess_features(features) ### normalize data matrix by row and transfer to sparsity vector form
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'BNF':
    support = adj  # Not used
    num_supports = 1
    model_func = BNF
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

##########      wz: change input to be a batch group, stack the features
# nsubj, nodes, signal_dim = features.shape()


# Define placeholders
placeholders = {
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)], # number of adj parametrized matrices
    'features': tf.placeholder(tf.float32, shape=(None,train_features.shape[1],train_features.shape[2])),
    'labels': tf.placeholder(tf.float32, shape=(None, 2)),
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=train_features.shape[2],output_dim=2,logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    # placeholders_test = placeholders
    # placeholders_test['features']=tf.placeholder(tf.float32, shape=features.shape)
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(train_features, support, train_labels, placeholders) # assign the first 4 values to placeholders

    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(test_features, support, test_labels, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# # Testing
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
