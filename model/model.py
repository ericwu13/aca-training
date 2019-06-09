from ops import *

VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, batchSize=4, dropout=0.5):
        self.var_dict = {}
        self.dropout = dropout
        self.batchSize = batchSize
        
        self.shapes = []
        self.layers = []
        self.layerNames = []
        self.args = []
        
        self.initShapes()

    def initShapes(self):
    
        rgb = tf.placeholder(tf.float32, [self.batchSize, 224*2, 224*2, 3])
        self.shapes.append(rgb.shape.as_list())
        
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224*2, 224*2, 1]
        assert green.get_shape().as_list()[1:] == [224*2, 224*2, 1]
        assert blue.get_shape().as_list()[1:] == [224*2, 224*2, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224*2, 224*2, 3]
        
        # layer 0
        conv1_1 = self.add_layer(bgr, self.conv_layer, 'conv1_1', 3, 64*2)
        
        # layer 1
        conv1_2 = self.add_layer(conv1_1, self.conv_layer, 'conv1_2', 64*2, 64*2)
        
        # layer 2        
        pool1 = self.add_layer(conv1_2, self.max_pool, 'pool1')
        
        # layer 3
        conv2_1 = self.add_layer(pool1, self.conv_layer, 'conv2_1', 64*2, 128*2)
        
        # layer 4
        conv2_2 = self.add_layer(conv2_1, self.conv_layer, 'conv2_2', 128*2, 128*2)
        
        # layer 5
        pool2 = self.add_layer(conv2_2, self.max_pool, 'pool2')
        
        # layer 6
        conv3_1 = self.add_layer(pool2, self.conv_layer, 'conv3_1', 128*2, 256*2)
        
        # layer 7
        conv3_2 = self.add_layer(conv3_1, self.conv_layer, 'conv3_2', 256*2, 256*2)
        
        # layer 8
        conv3_3 = self.add_layer(conv3_2, self.conv_layer, 'conv3_3', 256*2, 256*2)
        
        # layer 9
        conv3_4 = self.add_layer(conv3_3, self.conv_layer, 'conv3_4', 256*2, 256*2)
        
        # layer 10
        pool3 = self.add_layer(conv3_4, self.max_pool, 'pool3')
        
        # layer 11
        conv4_1 = self.add_layer(pool3, self.conv_layer, 'conv4_1', 256*2, 512*2)
        
        # layer 12
        conv4_2 = self.add_layer(conv4_1, self.conv_layer, 'conv4_2', 512*2, 512*2)
        
        # layer 13
        conv4_3 = self.add_layer(conv4_2, self.conv_layer, 'conv4_3', 512*2, 512*2)
        
        # layer 14
        conv4_4 = self.add_layer(conv4_3, self.conv_layer, 'conv4_4', 512*2, 512*2)

        # layer 15
        pool4 = self.add_layer(conv4_4, self.max_pool, 'pool4')
        
        # layer 16
        conv5_1 = self.add_layer(pool4, self.conv_layer, 'conv5_1', 512*2, 512*2)
        
        # layer 17
        conv5_2 = self.add_layer(conv5_1, self.conv_layer, 'conv5_2', 512*2, 512*2)

        # layer 18
        conv5_3 = self.add_layer(conv5_2, self.conv_layer, 'conv5_3', 512*2, 512*2)
        
        # layer 19
        conv5_4 = self.add_layer(conv5_3, self.conv_layer, 'conv5_4', 512*2, 512*2)
        
        # layer 20
        pool5 = self.add_layer(conv5_4, self.max_pool, 'pool5')

        # layer 21
        pool6 = self.add_layer(pool5, self.max_pool, 'pool6')
        
        # layer 22
        fc6 = self.add_layer(pool6, self.fc_layer, 'fc6', 25088*2, 4096)

        # layer 23
        relu6 = self.add_layer(fc6, self.relu_layer, 'relu6')
        
        # layer 24
        fc7 = self.add_layer(relu6, self.fc_layer, 'fc7', 4096, 4096)
        
        # layer 25
        relu7 = self.add_layer(fc7, self.relu_layer, 'relu7')
        
        # layer 26
        fc8 = self.add_layer(relu7, self.fc_layer, 'fc8', 4096, 1000)
        
        # layer 27
        prob = self.add_layer(fc8, tf.nn.softmax, 'prob')
        
        

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
            
    def relu_layer(self, input, name):
        return tf.nn.dropout(tf.nn.relu(input), self.dropout)

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        value = initial_value
        var = tf.Variable(value, name=var_name)
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        return var

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
       
    def add_layer(self, input, layer, name, *args):
        args = list(args)
        args.append(name)
        self.layers.append(layer)
        self.layerNames.append(name)
        self.args.append(args)
        if name == 'prob': output = layer(input, name=name)
        else: output = layer(input, *args)
        self.shapes.append(output.shape.as_list())
        return output
        
    def build_single_stage(self, i, j):   # this stage spans from layer i to j
        assert (0 <= i) and (i < j) and (j <= len(self.layers))
        tf.reset_default_graph()
        self.var_dict.clear()
        self.stgVars = []
        self.isLast = (j == len(self.layers))
        
        self.input = tf.placeholder(tf.float32, self.shapes[i])
        self._output = tf.placeholder(tf.float32, self.shapes[j])
        
        #if isTrain:
        #    self._input = tf.Variable(tf.zeros(self.shapes[i]), name='forplaceholder')
        #    _in = self._input.assign(self.input)
        #    self.tmp = _in
        #else:
        #    _in = self.input
        
        _in = self.input
        
        if i == 0:
            rgb_scaled = _in * 255.0
            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224*2, 224*2, 1]
            assert green.get_shape().as_list()[1:] == [224*2, 224*2, 1]
            assert blue.get_shape().as_list()[1:] == [224*2, 224*2, 1]
            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            assert bgr.get_shape().as_list()[1:] == [224*2, 224*2, 3]
            _in = bgr
        
        for k in range(i, j):
            name = self.layerNames[k]
            if name == 'prob':
                _in = self.layers[k](_in, name='prob')
            else:
                _in = self.layers[k](_in, *self.args[k])
            
            if name.startswith('fc') or name.startswith('conv'):
                for m in range(2): self.stgVars.append(self.var_dict[(name, m)])
            
        self.output = _in
        
    def single_stage_fp(self):
        pass
        
    def single_stage_bp(self):
        def myloss(g, a):
            return 0.5 * tf.math.multiply(tf.math.square(a), g)
        lossFn = tf.losses.softmax_cross_entropy if self.isLast else myloss
            
        loss = lossFn(self._output, self.output)
        opt = tf.train.GradientDescentOptimizer(0.001)
        #grad = opt.compute_gradients(loss, self.stgVars)
        #apply = opt.apply_gradients(grad)
        grad = tf.gradients(loss, self.stgVars+[self.input])
        apply = opt.apply_gradients(zip(grad[:-1], self.stgVars))
        #grad = [g[0] for g in grad[:2]]
        
        return grad[-1], apply
class ResNet(object):
    def __init__(self, batch_size):
        self.model_name = 'ResNet'

        self.img_size = 64
        self.c_dim = 3
        self.label_dim = 1000
        self.res_n = 152

        self.batch_size = batch_size
        self.shapes = []
        self.layers = []
        self.layerNames = []
        self.args = []
        
        self.initShapes()
        
    def initShapes(self):
        with tf.variable_scope("network", reuse=False):
        
            x = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim])
            self.shapes.append(x.shape.as_list())
            
            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock
            residual_list = get_residual_layer(self.res_n)
            ch = 64 # paper is 64
            
            x = self.add_layer(x, conv, 'conv', ch, 3, 1)

            for i in range(residual_list[0]):
                x = self.add_layer(x, residual_block, 'resblock0_'+str(i), ch, False)

            x = self.add_layer(x, residual_block, 'resblock1_0', ch*2, True)

            for i in range(1, residual_list[1]):
                x = self.add_layer(x, residual_block, 'resblock1_'+str(i), ch*2, False)
            
            x = self.add_layer(x, residual_block, 'resblock2_0', ch*4, True)

            for i in range(1, residual_list[2]):
                x = self.add_layer(x, residual_block, 'resblock2_'+str(i), ch*4, False)

            x = self.add_layer(x, residual_block, 'resblock3_0', ch*8, True)

            for i in range(1, residual_list[3]):
                x = self.add_layer(x, residual_block, 'resblock3_'+str(i), ch*8, False)

            x = self.add_layer(x, batch_norm, 'batch_norm')
            x = self.add_layer(x, relu, 'relu')
            x = self.add_layer(x, global_avg_pooling, 'global_avg_pooling')
            x = self.add_layer(x, fully_conneted, 'logit', self.label_dim)

            return x

    def add_layer(self, input, layer, name, *args):
        args = list(args)
        args.append(name)
        self.layers.append(layer)
        self.layerNames.append(name)
        self.args.append(args)
        output = layer(input, *args)
        self.shapes.append(output.shape.as_list())
        return output
        
    def build_single_stage(self, i, j):   # this stage spans from layer i to j
        assert (0 <= i) and (i < j) and (j <= len(self.layers))
        tf.reset_default_graph()
        self.isLast = (j == len(self.layers))
        
        self.input = tf.placeholder(tf.float32, self.shapes[i])
        self._output = tf.placeholder(tf.float32, self.shapes[j])
        
        _in = self.input
        
        for k in range(i, j):
            _in = self.layers[k](_in, *self.args[k])
            
        self.output = _in
        
    def single_stage_bp(self):
        def myloss(g, a):
            return 0.5 * tf.math.multiply(tf.math.square(a), g)
        lossFn = classification_loss if self.isLast else myloss
            
        loss = lossFn(self._output, self.output)
        opt = tf.train.GradientDescentOptimizer(0.001)
        vars = tf.trainable_variables()
        grad = tf.gradients(loss, vars+[self.input])
        apply = opt.apply_gradients(zip(grad[:-1], vars))
        
        return grad[-1], apply
