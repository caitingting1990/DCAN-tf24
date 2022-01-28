import tensorflow as tf
from tensorflow import keras
from myLayers import *
from loss import *
import random

baseNet=keras.Sequential([
    keras.layers.Conv1D(16,(64,),8,"same",input_shape=(1024,1)),
    DCCA(16,2),
    keras.layers.MaxPool1D((2,),2),

    keras.layers.Conv1D(32,(3,),1,"same"),
    DCCA(32,2),
    keras.layers.MaxPool1D((2,),2),

    keras.layers.Conv1D(64,(3,),1,"same"),
    DCCA(64,2),
    keras.layers.MaxPool1D((2,),2),

    keras.layers.Conv1D(64,(3,),1,"same"),
    DCCA(64,2),
    keras.layers.MaxPool1D((2,),2),

    keras.layers.Conv1D(64,(3,),1,"same"),
    DCCA(64,2),
    keras.layers.MaxPool1D((2,),2),

    keras.layers.Conv1D(64,(3,),1,"same"),   #[b,4,64]---gpool--[b,1,64]
    DCCA(64,2)
])

classifier=keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.Dense(10)
])

feature_corrector=DCFC(64)

output_corrector=DCFC(10)

class DCAN(keras.Model):
    def __init__(self):
        super(DCAN, self).__init__()
        self.baseNet=baseNet
        self.gpool=keras.layers.GlobalAveragePooling1D()
        self.feature_corrector=feature_corrector
        self.classifier=classifier
        self.output_corrector=output_corrector
    def call(self,inputs,training=None,mask=None):
        feature_base=self.baseNet(inputs)
        feature_correct=self.feature_corrector(feature_base)
        feature_correct=self.gpool(feature_correct)
        output_base=self.classifier(tf.expand_dims(feature_correct,axis=1))
        output_correct=self.output_corrector(output_base)
        softmax_output_correct=tf.nn.softmax(output_correct)
        return feature_correct,tf.squeeze(output_correct),tf.squeeze(softmax_output_correct)


