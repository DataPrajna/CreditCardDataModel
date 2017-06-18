<<<<<<< HEAD

import tensorflow as tf
import numpy as np

default_args = {
    'in_tnr': 'UN', #input tensor
     'w': 'UN',
     'b': 'UN',
     'cnv1_s': [1, 1, 1, 1],
     'MP_s': [1, 1, 2, 1], #Max_pooling
     'ks': [1, 1, 2, 1],   #Kernel size
     'conv_pad': 'SAME',
     'pool_pad': 'SAME',
     'out_name': 'UN',
     'is_relu': True,
     'stddev': 1e-1,
     'dtype': tf.float32,
     'shape': 'UN',
     'is_trainable': True,
     'const': 0,
     'do_flat': False,
     'is_pool': False
}


def set_w(args):
    shape_in = args['in_tnr'].shape.as_list()
    shape_out = args['shape']
    if len(shape_out) > 3 and shape_out[2] == -1:
        shape_out[2] = shape_in[-1]


    if len(shape_out) ==2 and shape_out[0] == -1:
        shape_out[0] = shape_in[-1]


    if args['do_flat']:
        width = int(np.prod(shape_in[1:]))
        shape_out[0] = width
    args['w'] = tf.Variable(tf.truncated_normal(shape=shape_out, dtype=tf.float32, stddev=args['stddev']), trainable=True,
                       name='weights')


def set_b(args):
    shape = [args['shape'][-1]]
    args['b'] = tf.Variable(tf.constant(args['const'], shape=shape, dtype=tf.float32), trainable=True, name='biases')

def update_params(args, vgg_params):
    var = vgg_params['var']
    set_w(args)
    set_b(args)
    var['{}.w'.format(args['out_name'])] = args['w']
    var['{}.b'.format(args['out_name'])] = args['b']


def conv_1d_node(args, vgg_params):
    with tf.name_scope(args['out_name']) as scope:
        update_params(args, vgg_params)
        tn = vgg_params['tn']
        conv_tnr = tf.nn.conv2d(args['in_tnr'], args['w'], strides=args['cnv1_s'], padding=args['conv_pad'])
        conv_tnr = tf.nn.bias_add(conv_tnr, args['b'])

        if args['is_relu']:
            conv_tnr = tf.nn.relu(conv_tnr, name=scope)

        if args['is_pool']:
            conv_tnr = tf.nn.max_pool(conv_tnr, ksize=args['ks'], strides=args['MP_s'], padding=args['pool_pad'],
                           name=args['out_name'])
        tn[args['out_name']] = conv_tnr



def fc_node(args, vgg_params):
    with tf.name_scope(args['out_name']) as scope:
        update_params(args, vgg_params)
        tn = vgg_params['tn']
        if args['do_flat']:
            height = np.prod(args['in_tnr'].shape.as_list()[1:])
            in_tnr = tf.reshape(args['in_tnr'], [-1, height])
            fc_tn = tf.nn.bias_add(tf.matmul(in_tnr, args['w']), args['b'])
        else:
            fc_tn = tf.nn.bias_add(tf.matmul(args['in_tnr'], args['w']), args['b'])
        if args['is_relu']:
           fc_tn = tf.nn.relu(fc_tn)
        tn[args['out_name']] = fc_tn


def build_vgg16(init_node=None):
    vgg_graph = {'tn': dict(),
                 'var': dict()}

    tn = vgg_graph['tn']
    var = vgg_graph['var']
    args = dict()
    var['data_mean'] = tf.Variable(
    tf.constant([0.0], dtype=tf.float32, shape=[1, 1, 1, 1], name='data_mean'), trainable=False)

    if init_node == None:
        init_node = tf.placeholder(tf.float32, [None, 1,  400, 1])

    tn['place_holder'] = init_node
    tn['init_node'] = tn['place_holder'] - var['data_mean']

    args.update(default_args, in_tnr=tn['init_node'], shape=[1, 3, 1, 64], out_name='conv_1_1')
    conv_1d_node(args, vgg_graph)
    args.update(default_args, in_tnr=tn['conv_1_1'], shape=[1, 3, -1, 64], out_name='conv_1_2', is_pool=True)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_1_2'], shape=[1, 3, -1, 64], out_name='conv_2_1')
    conv_1d_node(args, vgg_graph)
    args.update(default_args, in_tnr=tn['conv_2_1'], shape=[1, 3, -1, 64], out_name='conv_2_2', is_pool=True)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_2_2'], shape=[1, 3, -1, 64], out_name='conv_3_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_1'], shape=[1, 3, -1, 64], out_name='conv_3_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_2'], shape=[1, 3, -1, 64], out_name='conv_3_3', is_pool=False)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_3'], shape=[1, 3, -1, 64], out_name='conv_4_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_1'], shape=[1, 3, -1, 64], out_name='conv_4_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_2'], shape=[1, 3, -1, 64], out_name='conv_4_3', is_pool=False)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_3'], shape=[1, 3, -1, 64], out_name='conv_5_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_1'], shape=[1, 3, -1, 64], out_name='conv_5_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_2'], shape=[1, 3, -1, 64], out_name='conv_5_3', is_pool=False, padding = 'VALID')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_3'], shape=[-1,200], out_name='fc_1',do_flat=True)
    fc_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['fc_1'], shape=[-1, 400], out_name='fc_2')
    fc_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['fc_2'], shape=[-1, 2], out_name='fc_3', is_relu=False)
    fc_node(args, vgg_graph)

    return vgg_graph









if __name__=='__main__':
    imgs = tf.placeholder(tf.float32, [None, 400, 1])
    sess = tf.Session()
    vgg_graph = build_vgg16()
    print(vgg_graph['var']['fc_3.w'].shape)

=======

import tensorflow as tf
import numpy as np

default_args = {
    'in_tnr': 'UN', #input tensor
     'w': 'UN',
     'b': 'UN',
     'cnv1_s': [1, 1, 1, 1],
     'MP_s': [1, 1, 2, 1], #Max_pooling
     'ks': [1, 1, 2, 1],   #Kernel size
     'conv_pad': 'SAME',
     'pool_pad': 'SAME',
     'out_name': 'UN',
     'is_relu': True,
     'stddev': 1e-1,
     'dtype': tf.float32,
     'shape': 'UN',
     'is_trainable': True,
     'const': 0,
     'do_flat': False,
     'is_pool': False
}


def set_w(args):
    shape_in = args['in_tnr'].shape.as_list()
    shape_out = args['shape']
    if len(shape_out) > 3 and shape_out[2] == -1:
        shape_out[2] = shape_in[-1]


    if len(shape_out) ==2 and shape_out[0] == -1:
        shape_out[0] = shape_in[-1]


    if args['do_flat']:
        width = int(np.prod(shape_in[1:]))
        shape_out[0] = width
    args['w'] = tf.Variable(tf.truncated_normal(shape=shape_out, dtype=tf.float32, stddev=args['stddev']), trainable=True,
                       name='weights')


def set_b(args):
    shape = [args['shape'][-1]]
    args['b'] = tf.Variable(tf.constant(args['const'], shape=shape, dtype=tf.float32), trainable=True, name='biases')

def update_params(args, vgg_params):
    var = vgg_params['var']
    set_w(args)
    set_b(args)
    var['{}.w'.format(args['out_name'])] = args['w']
    var['{}.b'.format(args['out_name'])] = args['b']


def conv_1d_node(args, vgg_params):
    with tf.name_scope(args['out_name']) as scope:
        update_params(args, vgg_params)
        tn = vgg_params['tn']
        conv_tnr = tf.nn.conv2d(args['in_tnr'], args['w'], strides=args['cnv1_s'], padding=args['conv_pad'])
        conv_tnr = tf.nn.bias_add(conv_tnr, args['b'])

        if args['is_relu']:
            conv_tnr = tf.nn.relu(conv_tnr, name=scope)

        if args['is_pool']:
            conv_tnr = tf.nn.max_pool(conv_tnr, ksize=args['ks'], strides=args['MP_s'], padding=args['pool_pad'],
                           name=args['out_name'])
        tn[args['out_name']] = conv_tnr



def fc_node(args, vgg_params):
    with tf.name_scope(args['out_name']) as scope:
        update_params(args, vgg_params)
        tn = vgg_params['tn']
        if args['do_flat']:
            height = np.prod(args['in_tnr'].shape.as_list()[1:])
            in_tnr = tf.reshape(args['in_tnr'], [-1, height])
            fc_tn = tf.nn.bias_add(tf.matmul(in_tnr, args['w']), args['b'])
        else:
            fc_tn = tf.nn.bias_add(tf.matmul(args['in_tnr'], args['w']), args['b'])
        if args['is_relu']:
           fc_tn = tf.nn.relu(fc_tn)
        tn[args['out_name']] = fc_tn


def build_vgg16(init_node=None):
    vgg_graph = {'tn': dict(),
                 'var': dict()}

    tn = vgg_graph['tn']
    var = vgg_graph['var']
    args = dict()
    var['data_mean'] = tf.Variable(
    tf.constant([0.0], dtype=tf.float32, shape=[1, 1, 1, 1], name='data_mean'), trainable=False)

    if init_node == None:
        init_node = tf.placeholder(tf.float32, [None, 1,  400, 1])

    tn['place_holder'] = init_node
    tn['init_node'] = tn['place_holder'] - var['data_mean']

    args.update(default_args, in_tnr=tn['init_node'], shape=[1, 3, 1, 64], out_name='conv_1_1')
    conv_1d_node(args, vgg_graph)
    args.update(default_args, in_tnr=tn['conv_1_1'], shape=[1, 3, -1, 64], out_name='conv_1_2', is_pool=True)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_1_2'], shape=[1, 3, -1, 64], out_name='conv_2_1')
    conv_1d_node(args, vgg_graph)
    args.update(default_args, in_tnr=tn['conv_2_1'], shape=[1, 3, -1, 64], out_name='conv_2_2', is_pool=True)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_2_2'], shape=[1, 3, -1, 64], out_name='conv_3_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_1'], shape=[1, 3, -1, 64], out_name='conv_3_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_2'], shape=[1, 3, -1, 64], out_name='conv_3_3', is_pool=False)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_3_3'], shape=[1, 3, -1, 64], out_name='conv_4_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_1'], shape=[1, 3, -1, 64], out_name='conv_4_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_2'], shape=[1, 3, -1, 64], out_name='conv_4_3', is_pool=False)
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_4_3'], shape=[1, 3, -1, 64], out_name='conv_5_1')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_1'], shape=[1, 3, -1, 64], out_name='conv_5_2')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_2'], shape=[1, 3, -1, 64], out_name='conv_5_3', is_pool=False, padding = 'VALID')
    conv_1d_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['conv_5_3'], shape=[-1,200], out_name='fc_1',do_flat=True)
    fc_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['fc_1'], shape=[-1, 400], out_name='fc_2')
    fc_node(args, vgg_graph)

    args.update(default_args, in_tnr=tn['fc_2'], shape=[-1, 2], out_name='fc_3', is_relu=False)
    fc_node(args, vgg_graph)

    return vgg_graph









if __name__=='__main__':
    imgs = tf.placeholder(tf.float32, [None, 400, 1])
    sess = tf.Session()
    vgg_graph = build_vgg16()
    print(vgg_graph['var']['fc_3.w'].shape)

>>>>>>> 59c742463e2ba8cf8736349ba00314e76d891611
