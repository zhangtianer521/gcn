from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            # if self.logging and not self.sparse_inputs:
            #     tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            # if self.logging:                tf.summary.histogram(self.name + '/outputs', outputs)

            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])




class Batch_GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Batch_GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'): #### initialize the weights and bias for the adj (layers)
            for i in range(len(self.support)):  ##### only use one-localize
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  ### input: [xn, xm, xk]; output: [xn, xm, output_dim]
        x = inputs

        xn, xm, xk = x.get_shape()

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                #### batch only useful to non-sparse inputs, dim(pre_sup) = [xn*xm,output_dim], T is the filters number
                x=tf.reshape(x,[xn*xm,xk])
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            #### transpost pre_sup to have proper left multiplication
            # _, _, nfilter = pre_sup.get_shape()
            pre_sup = tf.reshape(pre_sup,[xn, xm, self.output_dim])
            pre_sup = tf.transpose(pre_sup, perm=[1,2,0]) #### [xm, output_dim, xn]
            pre_sup = tf.reshape(pre_sup,[xm, xn*self.output_dim]) ### [xm, xn*output_dim]
            support = dot(self.support[i], pre_sup, sparse=False) ### [xm, xn*output_dim]
            support = tf.reshape(support,[xm, self.output_dim, xn])
            support = tf.transpose(support,perm=[2,0,1])  ### back to [xn, xm, nfilteroutput_dim]
            supports.append(support)
        output = tf.add_n(supports)  #### if working on K-neighborhood, there will be multiple supports

        # bias
        if self.bias:
            output = tf.reshape(output, [xn*xm, self.output_dim])
            output += self.vars['bias']
            output = tf.reshape(output,[xn, xm, self.output_dim])

        return self.act(output)

class Batch_Dense(Layer): ### node wise dense layer
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Batch_Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, self.output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  ### input: [xn, xm, xk]; output: [xn, xm, output_dim]
        x = inputs

        xn, xm, xk = x.get_shape()

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        x = tf.reshape(x,[xn*xm, xk])
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs) ### [xn*xm, output_dim]

        # bias
        if self.bias:
            output += self.vars['bias']

        output = tf.reshape(output,[xn,xm,self.output_dim])
        return self.act(output)

class Batch_FC(Layer): ### graph wise dense layer
    """Dense layer."""
    def __init__(self, input_dim, output_dim, node_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Batch_FC, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([node_dim*input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  ### input: [xn, xm, xk]; output: [xn, output_dim]
        x = inputs

        xn, xm, xk = x.get_shape()

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        x = tf.reshape(x,[xn, xm * xk])
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs) ### [xn, output_dim]

        # bias
        if self.bias:
            output += self.vars['bias']

        # output = tf.reshape(output,[xn,output_dim])
        return self.act(output)
