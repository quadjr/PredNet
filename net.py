import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable

class EltFilter(chainer.Link):
    def __init__(self, width, height, channels, batchSize = 1, wscale=1, bias=0, nobias=False,
                initialW=None, initial_bias=None):
        W_shape = (batchSize, channels, height, width)
        super(EltFilter, self).__init__(W=W_shape)
        
        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * numpy.sqrt(1. / (width * height * channels))
            self.W.data[...] = numpy.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', W_shape)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias
            
    def __call__(self, x):
        y = x * self.W
        if self.b is not None:
            y = y + self.b
        return y

class ConvLSTM(chainer.Chain):
    def __init__(self, width, height, in_channels, out_channels, batchSize = 1):
        self.state_size = (batchSize, out_channels, height, width)
        self.in_channels = in_channels
        super(ConvLSTM, self).__init__(
            h_i=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_i=EltFilter(width, height, out_channels, nobias=True),
            
            h_f=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_f=EltFilter(width, height, out_channels, nobias=True),

            h_c=L.Convolution2D(out_channels, out_channels, 3, pad=1),

            h_o=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_o=EltFilter(width, height, out_channels, nobias=True),
        )

        for nth in range(len(self.in_channels)):
            self.add_link('x_i' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_f' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_c' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_o' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            
        self.reset_state()

    def to_cpu(self):
        super(ConvLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x):
        if self.h is None:
            self.h = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')
        if self.c is None:
            self.c = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')
        
        
        ii = self.x_i0(x[0])
        for nth in range(1, len(self.in_channels)):
            ii += getattr(self, 'x_i' + str(nth))(x[nth])
        ii += self.h_i(self.h)
        ii += self.c_i(self.c)
        ii = F.sigmoid(ii)

        ff = self.x_f0(x[0])
        for nth in range(1, len(self.in_channels)):
           ff += getattr(self, 'x_f' + str(nth))(x[nth])
        ff += self.h_f(self.h)
        ff += self.c_f(self.c)
        ff = F.sigmoid(ff)

        cc = self.x_c0(x[0])
        for nth in range(1, len(self.in_channels)):
           cc += getattr(self, 'x_c' + str(nth))(x[nth])
        cc += self.h_c(self.h)
        cc = F.tanh(cc)
        cc *= ii
        cc += (ff * self.c)

        oo = self.x_o0(x[0])
        for nth in range(1, len(self.in_channels)):
           oo += getattr(self, 'x_o' + str(nth))(x[nth])
        oo += self.h_o(self.h)
        oo += self.c_o(self.c)
        oo = F.sigmoid(oo)
        y = oo * F.tanh(cc)

        self.c = cc
        self.h = y
        return y

class PredNet(chainer.Chain):
    def __init__(self, width, height, channels, r_channels = None, batchSize = 1):
        super(PredNet, self).__init__()
        if r_channels is None:
            r_channels = channels
        
        self.layers = len(channels)
        self.sizes = [None]*self.layers
        w,h = width, height
        for nth in range(self.layers):
            self.sizes[nth] = (batchSize, channels[nth], h, w)
            w = w / 2
            h = h / 2
        
        for nth in range(self.layers):
            if nth != 0:
                self.add_link('ConvA' + str(nth), L.Convolution2D(channels[nth - 1] *2, channels[nth], 3, pad=1))
            
            self.add_link('ConvP' + str(nth), L.Convolution2D(r_channels[nth], channels[nth], 3, pad=1))
            
            if nth == self.layers - 1:
                self.add_link('ConvLSTM' + str(nth), ConvLSTM(self.sizes[nth][3], self.sizes[nth][2],
                               (self.sizes[nth][1] * 2, ), r_channels[nth]))
            else:
                self.add_link('ConvLSTM' + str(nth), ConvLSTM(self.sizes[nth][3], self.sizes[nth][2],
                               (self.sizes[nth][1] * 2, r_channels[nth + 1]), r_channels[nth]))
                
        self.reset_state()

    def to_cpu(self):
        super(PredNet, self).to_cpu()
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        super(PredNet, self).to_gpu(device)
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_gpu(device)

    def reset_state(self):
        for nth in range(self.layers):
            setattr(self, 'P' + str(nth), None)
            getattr(self, 'ConvLSTM' + str(nth)).reset_state()

    def __call__(self, x):
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is None:
                setattr(self, 'P' + str(nth), variable.Variable(
                    self.xp.zeros(self.sizes[nth], dtype=x.data.dtype),
                    volatile='auto'))

        E = [None] * self.layers
        for nth in range(self.layers):
            if nth == 0:
                E[nth] = F.concat((F.relu(x - getattr(self, 'P' + str(nth))),
                                  F.relu(getattr(self, 'P' + str(nth)) - x)))
            else:
                A = F.max_pooling_2d(F.relu(getattr(self, 'ConvA' + str(nth))(E[nth - 1])), 2, stride = 2)
                E[nth] = F.concat((F.relu(A - getattr(self, 'P' + str(nth))),
                                  F.relu(getattr(self, 'P' + str(nth)) - A)))

        R = [None] * self.layers
        for nth in reversed(range(self.layers)):
            if nth == self.layers - 1:
                R[nth] = getattr(self, 'ConvLSTM' + str(nth))((E[nth],))
            else:
                upR = F.unpooling_2d(R[nth + 1], 2, stride = 2, cover_all=False)
                R[nth] = getattr(self, 'ConvLSTM' + str(nth))((E[nth], upR))

            if nth == 0:
                setattr(self, 'P' + str(nth), F.clipped_relu(getattr(self, 'ConvP' + str(nth))(R[nth]), 1.0))
            else:
                setattr(self, 'P' + str(nth), F.relu(getattr(self, 'ConvP' + str(nth))(R[nth])))
        
        return self.P0
