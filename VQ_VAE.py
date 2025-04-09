import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ee
import geemap
import random
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VectorQuantizer(layers.Layer):

  def __init__(self, num_embeddings, embedding_dim, input_dim, **kwargs):
    super().__init__(**kwargs)

    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.embeddings = tf.Variable(
      initial_value=tf.random.uniform([self.embedding_dim, self.num_embeddings], 0.0, 1.0),
      trainable=True, name="embeddings_vqvae"
    )

    self.input_dim = input_dim
    self.inx_embeddings = layers.Embedding(self.input_dim, 2) # for decoder

  def call(self, x):

    distances = ( tf.reduce_sum(x ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0)
        - 2 * tf.matmul(x, self.embeddings) )
    encoding_indices = tf.argmin(distances, axis=1)
    encodings = tf.one_hot(encoding_indices, self.num_embeddings)
    quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

    commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
    codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
    self.add_loss(commitment_loss*0.25 + codebook_loss)

    quantized = x + tf.stop_gradient(quantized - x)

    return quantized

class VQ_VAE:

  def __getExprsion(self, hidSize, inDim):

    big_exp = ''

    for x1 in range(hidSize):

      exp1 = ' '.join(['b("b'+str(i1)+'")*w'+str(x1)+'_'+str(i1)+' +' for i1 in range(inDim)])[:-2]
      exp1 = 'wl'+str(x1)+' * max('+exp1+' + b'+str(x1)+', 0)'

      big_exp = big_exp + exp1 + ' + ' + '\n'

    big_exp = big_exp[:-2] +  ' bl'

    return big_exp

  def __getWDict(self, mat1, mat2, mat3, bl):

    big_dict = {}

    '''For Hidden Layer'''
    for x1 in range(mat1.shape[0]):
      for x2 in range(mat1.shape[1]):
        big_dict['w'+str(x1)+'_'+str(x2)] = float(mat1[x1,x2])

      big_dict['b'+str(x1)] = float(mat2[x1])

    '''For Last Layer'''
    for x1 in range(mat3.shape[0]):
      big_dict['wl'+str(x1)] = float(mat3[x1])

    big_dict['bl'] = float(bl)

    return big_dict

  def __init__(self, dataCSVPath, scalingPerc, nClusters, ndviStack, epochs):

    self.nClusters = nClusters

    df = pd.read_csv(dataCSVPath, index_col=None, header=0)
    self.colNames = list(df.columns.values)

    x_train = df.to_numpy()

    pc1 = np.percentile(x_train, scalingPerc[0])
    pc2 = np.percentile(x_train, scalingPerc[1])
    x_train = (x_train - pc1) / (pc2 - pc1)
    x_train[x_train > 1] = 1
    x_train[x_train < 0] = 0

    input_dim = x_train.shape[1]
    latent_dim = 1
    num_embeddings = nClusters # number of classes
    data_var = np.var(x_train)

    x_in = keras.Input(shape=(input_dim,))
    x_t = layers.Dense(128, activation="relu")(x_in)
    x_e = layers.Dense(latent_dim, activation="relu")(x_t)

    self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, input_dim)
    x_q = self.vq_layer(x_e)

    x_t = layers.Dense(256, activation="relu")(x_q)
    x_t = layers.Dense(256, activation="relu")(x_t)
    x_out = layers.Dense(input_dim, activation="sigmoid")(x_t)

    self.model1 = keras.Model(x_in, x_out)

    def custom_loss(y_true, y_pred):
      mse = tf.reduce_mean(tf.square(y_true - y_pred)) / data_var
      total_loss = mse +  tf.reduce_mean(self.vq_layer.losses)
      return total_loss

    self.model1.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss)

    self.fHist = self.model1.fit(x_train, x_train, epochs=epochs, batch_size=512, validation_split=0.2,verbose=0)

    self.model1_d = keras.Model(x_q, x_out)

    w1 = np.transpose(self.model1.layers[1].weights[0].numpy())
    b1 = self.model1.layers[1].weights[1].numpy()
    w2 = self.model1.layers[2].weights[0].numpy()[:,0]
    b2 = self.model1.layers[2].weights[1].numpy()
    emb = self.vq_layer.embeddings.numpy()[0,:]

    ndviStack_scl = ndviStack.subtract(pc1).divide(pc2 - pc1) # scalling image bands
    ndviStack_scl = ndviStack_scl.where(ndviStack_scl.gt(1), 1)
    ndviStack_scl = ndviStack_scl.where(ndviStack_scl.lt(0), 0)

    ndviStack_scl = ndviStack_scl.select(ndviStack.bandNames().getInfo(), ['b'+str(ii) for ii in range(input_dim)])

    ndviStack_pred = ndviStack_scl.expression(self.__getExprsion(128, input_dim), self.__getWDict(w1, b1, w2, b2))
    ndviStack_pred = ndviStack_pred.expression('1 / (1 + exp(-1*x0))', {'x0': ndviStack_pred.select('constant') })

    dist_list = [ndviStack_pred.subtract(ee.Image.constant(float(v))).abs().multiply(-1) for v in list(emb)]
    self.ndviStack_pred_argmin = ee.Image(dist_list).toArray().arrayArgmax().arrayGet(0)

  def plotLossCurve(self):
    
    plt.plot(self.fHist.history['loss'])
    plt.plot(self.fHist.history['val_loss'])
    plt.title('Model Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

  def plotClusterCentersVQVAE(self):
    
    for i1 in range(self.nClusters):
      cluster_id = i1

      vq_val = self.vq_layer.embeddings.numpy()[0,cluster_id].reshape((1,1))
      d_pred = self.model1_d.predict(vq_val, verbose=0)

      plt.plot(self.colNames, d_pred[0,:], label=f'Cluster {cluster_id+1}')

    plt.legend(loc='lower right')
    plt.xlabel("Year")
    plt.ylabel("NDVI")
    plt.title("Cluster Centers")
    plt.xticks(rotation='vertical')
    plt.show()

  def getClusteredResult(self):

    return self.ndviStack_pred_argmin 