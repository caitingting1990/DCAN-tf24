import tensorflow as tf

def EntropyLoss(input_):
    mask = input_>=0.0000001
    mask_out = input_[mask]
    entropy = - (tf.math.reduce_sum(mask_out * tf.math.log(mask_out)))
    return entropy / float(input_.shape[0])


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = tf.concat([source, target], axis=0)
    tmp_shape=(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    total0=tf.expand_dims(total,axis=0)
    total0 = tf.broadcast_to(total0,shape=tmp_shape)
    total1=tf.expand_dims(total,axis=1)
    total1 = tf.broadcast_to(total1,shape=tmp_shape)
    L2_distance = tf.math.cumsum((total0 - total1) ** 2,axis=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    res=sum(kernel_val)
    return res  # /len(kernel_val)

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def MMD_reg(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_source = int(source.shape[0])
    batch_size_target = int(target.shape[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size_source):
        s1, s2 = i, (i + 1) % batch_size_source
        t1, t2 = s1 + batch_size_target, s2 + batch_size_target
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size_source + batch_size_target)