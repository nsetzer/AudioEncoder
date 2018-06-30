

class lstm2(Model):
    """docstring for autoencoder"""
    def __init__(self, **kwargs):
        super(autoencoder, self).__init__()

        self.settings = {
            "nClasses": None
            "nRecurrentUnits": 100,
        }
        self.dimensions = kwargs

        self.updateSettings()

    def __call__(self, x, y, reuse=False, isTraining=False):

        n_classes = settings['nClasses']
        n_hidden = settings['nRecurrentUnits']

        wshape = [n_hidden, n_classes]
        weight = tf.Variable(tf.truncated_normal(wshape, stddev = 0.1))

        bshape = [n_classes]
        bias = tf.Variable(tf.constant(0.0, shape = bshape))

        cell = rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype = tf.float32)
        output_flattened = tf.reshape(output, [-1, n_hidden])
        output_logits = tf.add(tf.matmul(output_flattened,weight),bias)
        output_all = tf.nn.sigmoid(output_logits)
        output_reshaped = tf.reshape(output_all,[-1,n_steps,n_classes])
        output_last = tf.gather(tf.transpose(output_reshaped,[1,0,2]), n_steps - 1)
        #output = tf.transpose(output, [1, 0, 2])
        #last = tf.gather(output, int(output.get_shape()[0]) - 1)
        #output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
        return output_logits, output_last, output_all