import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from loss import *
from myModel import *
import numpy as np
import random

BATCH_SIZE=128 #总样本数128
CLASS_NUM=10
alpha=1.5
beta=0.1
gama=1/CLASS_NUM
mnist=tf.keras.datasets.mnist
(x,y),(x_,y_)=mnist.load_data()
x=np.reshape(x,(-1,784))/255.
x_=np.reshape(x_,(-1,784))/255.
pad1=np.zeros(shape=(60000,240))
pad2=np.zeros(shape=(10000,240))
inputs=np.concatenate((x,pad1),axis=1)
tests=np.concatenate((x_,pad2),axis=1)
train_db=tf.data.Dataset.from_tensor_slices((inputs,y)).shuffle(10000).batch(BATCH_SIZE)
test_db=tf.data.Dataset.from_tensor_slices((tests,y_)).repeat().batch(BATCH_SIZE)
# [b,1024,1]
sample=next(iter(train_db))

myModel=DCAN()
myModel.build(input_shape=(256,1024,1))
optimizer=tf.optimizers.Adam(learning_rate=1e-3)

def reg_loss(train_label,feature_correct, output_correct):
    sum_reg_loss = 0
    labels=train_label[:BATCH_SIZE,].numpy()
    for k in range(CLASS_NUM):
        source_k_index = []
        for index, source_k in enumerate(labels):
            # find all indexes of k-th class source samples
            if source_k == k:
                source_k_index.append(index)
        fea_reg_loss = 0
        out_reg_loss = 0
        if len(source_k_index) > 0:
            # random subset indexes of source samples
            source_rand_index = []
            index = 0
            for z in range(BATCH_SIZE):
                prob = random.random()
                if prob < 0.8/CLASS_NUM:
                    source_rand_index.append(index)
                    index += 1

            if len(source_rand_index) > 0:
                # source feature of k-th class
                source_k_fea = tf.gather(feature_correct,source_k_index,axis=0)
                source_k_out = tf.gather(output_correct,source_k_index,axis=0)

                # random selected source feature
                source_rand_fea = tf.gather(feature_correct, source_rand_index, axis=0)
                source_rand_out = tf.gather(output_correct, source_rand_index, axis=0)

                fea_reg_loss = MMD_reg(source_k_fea, source_rand_fea)
                out_reg_loss = MMD_reg(source_k_out, source_rand_out, kernel_num=1, fix_sigma=1.68)
        sum_reg_loss += (tf.reduce_mean(fea_reg_loss).numpy()+tf.reduce_mean(out_reg_loss).numpy())
    return tf.cast(sum_reg_loss,tf.float32)

def batch_acc(y_true,softmax_pred):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(softmax_pred, axis=1),
                     tf.cast(y_true, dtype=tf.int64)), tf.float32)
    )


for epoch in range(5):
    test_iter = iter(test_db)
    for step,(train_data,train_label) in enumerate(train_db):
        with tf.GradientTape() as tape:
            tape.watch(myModel.trainable_variables)
            test_data=next(test_iter)
            test_data=test_data if test_data[0].shape[0]==BATCH_SIZE else next(test_iter)
            input_data=tf.concat([train_data,test_data[0]],axis=0)    #[2b,1024,1]
            input_label=tf.concat([train_label,test_data[1]],axis=0)  #[2b,1]
            feature_correct, output_correct, softmax_output_correct=myModel(input_data,training=True)
            classify_loss = tf.losses.sparse_categorical_crossentropy(input_label[:BATCH_SIZE, ],
                                                                      softmax_output_correct[:BATCH_SIZE],
                                                                      from_logits=False)
            entropy_loss = EntropyLoss(softmax_output_correct[BATCH_SIZE:, ])

            transfer_loss = tf.reduce_mean(MMD(feature_correct[:BATCH_SIZE, ], feature_correct[BATCH_SIZE:, ]))
            transfer_loss += tf.reduce_mean(
                MMD(output_correct[:BATCH_SIZE, ], output_correct[BATCH_SIZE:, ]))
            sum_res_loss = reg_loss(input_label, feature_correct, output_correct)

            train_source_acc = batch_acc(input_label[:BATCH_SIZE,],softmax_output_correct[:BATCH_SIZE])
            train_target_acc = batch_acc(input_label[BATCH_SIZE:, ], softmax_output_correct[BATCH_SIZE:])

            # 总损失=源域分类损失 + alpha*域间特征对齐修正损失 + beta*目标域分类熵损失 + gama * 源域内部分布对齐修正损失
            total_loss=classify_loss+alpha*transfer_loss+beta*entropy_loss+gama*sum_res_loss

        grads=tape.gradient(total_loss,myModel.trainable_variables)
        optimizer.apply_gradients(zip(grads,myModel.trainable_variables))
        if step % 10==0:
            (test_data,test_label)=next(test_iter)
            (test_data,test_label) = (test_data,test_label) if test_data.shape[0]== BATCH_SIZE else next(test_iter)
            (_,_,test_pred)=myModel(test_data,training=False)
            test_accuracy = batch_acc(test_label,test_pred)
            print("epoch:%d,step:%d,total_loss:%f,source_acc:%f,target_acc:%f,test_acc:%f"
                  %(epoch,step,tf.reduce_mean(total_loss),train_source_acc,train_target_acc,test_accuracy))
        if epoch % 5==0:
            myModel.save_weights("./mymodel_weights")

