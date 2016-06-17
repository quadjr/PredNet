import argparse
import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net

parser = argparse.ArgumentParser(
description='PredNet')
parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

#Create Model
inWidth = 160
inHeight = 128
inChannel = 3
prednet = net.PredNet(inWidth, inHeight, (inChannel,48,96,192))
model = L.Classifier(prednet, lossfun=mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
#    print(str(image.shape[0])+'x'+str(image.shape[1])+'x'+str(image.shape[2]))
    top = (image.shape[1]  - inHeight) / 2
    left = (image.shape[2]  - inWidth) / 2
    bottom = inHeight + top
    right = inWidth + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image /= 255
    return image

def writeImage(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

imagelist = load_image_list(args.train, args.root)


for num in range(0, 10000):
    bprop_len = 20
    prednet.reset_state()
    model.zerograds()
    loss = 0

    batchSize = 1
    x_batch = np.ndarray((batchSize, inChannel, inHeight, inWidth), dtype=np.float32)
    y_batch = np.ndarray((batchSize, inChannel, inHeight, inWidth), dtype=np.float32)
    x_batch[0] = read_image(imagelist[0]);
    for i in range(1, len(imagelist)):
        y_batch[0] = read_image(imagelist[i]);
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))

        print('i:' + str(i))
        if (i + 1) % bprop_len == 0:
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            loss = 0
            optimizer.update()
            model.to_cpu()
            writeImage(x_batch[0].copy(), 'out/' + str(num) + '_' + str(i) + 'a.jpg')
            writeImage(model.y.data[0].copy(), 'out/' + str(num) + '_' + str(i) + 'b.jpg')
            writeImage(y_batch[0].copy(), 'out/' + str(num) + '_' + str(i) + 'c.jpg')
            model.to_gpu()
            print('loss:' + str(float(model.loss.data)))

        if i == 1 and (num%10) == 0:
            print('save the model')
            serializers.save_npz('out/' + str(num) + '.model', model)
            print('save the optimizer')
            serializers.save_npz('out/' + str(num) + '.state', optimizer)

        x_batch[0] = y_batch[0]
