from ops import *

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

