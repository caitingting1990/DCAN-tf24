import tensorflow as tf
from tensorflow import keras

class DCCA(keras.layers.Layer):
    @tf.autograph.experimental.do_not_convert
    def __init__(self,channel,reduction):
        super(DCCA, self).__init__()
        self.gpool=keras.layers.GlobalAveragePooling1D()
        self.fc0=keras.layers.Conv1D(channel//reduction,(1,),1,'same')
        self.fc1=keras.layers.Conv1D(channel//reduction,(1,),1,'same')
        self.relu=keras.layers.Activation("relu")
        self.fc2=keras.layers.Conv1D(channel,(1,),1,'same')
        self.sigmoid=keras.layers.Activation("sigmoid")
        pass
    def call(self,inputs,training=None):
        ## inputs [2b,h,w,c] ----[2b,n,c] <=S[b,n,c] T[b,n,c]
        if inputs.shape[0] is not None:
            batch_size=inputs.shape[0]//2
        else:
            batch_size=64
        x=self.gpool(inputs)
        x=tf.expand_dims(x,axis=1)  #--[2b,1,c]
        if training:
            src = x[:batch_size, :] #[b,1,c]
            src=self.fc0(src)       #[b,1,c/r]
            tar = x[batch_size:, :] #[b,1,c]
            tar=self.fc1(tar)       #[b,1,c/r]
            x=tf.concat((src,tar),axis=0) #[2b,1,c/r]
        else:
            x=self.fc1(x)           #[2b,1,c/r]
        x=self.relu(x)
        x=self.fc2(x)               #[2b,1,c]
        x=self.sigmoid(x)
        return inputs*x

class DCFC(keras.layers.Layer):
    def __init__(self,channel):
        super(DCFC, self).__init__()
        self.fc0=keras.layers.Conv1D(channel,(1,),1,'same')
        self.fc1=keras.layers.Conv1D(channel,(1,),1,'same')
        pass
    @tf.autograph.experimental.do_not_convert
    def call(self,inputs,training=None):
        if inputs.shape[0] is not None:
            batch_size = inputs.shape[0] // 2
        else:
            batch_size = 64
        x = inputs                #[2b,n,c]
        if training:
            src = x[:batch_size,] #[b,n,c]
            tar = x[batch_size:,] #[b,n,c]
            res_tar=self.fc0(tar) #[b,n,c]
            res_tar=self.fc1(res_tar) #[b,n,c]
            tar=tf.add(tar,res_tar)
            x=tf.concat((src,tar),axis=0) #[2b,n,c]
        else:
            res_x=self.fc0(x)          #[2b,n,c]
            res_x=self.fc1(res_x)      #[2b,n,c]
            x=tf.add(x,res_x)          #[2b,n,c]
        return x