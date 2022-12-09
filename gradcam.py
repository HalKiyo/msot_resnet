import cv2
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def image_preprocess(val, index):
    img = val.copy()
    x = img[index]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def grad_cam(input_model, image, y_val, layer_name):
    pred_val = input_model.output[0]
    y_val = tf.convert_to_tensor(y_val.astype(np.float32))
    loss = K.mean(K.square(pred_val - y_val))
    conv_output = input_model.get_layer(layer_name).output
    #---2. gradient from loss to last conv layer
    grads = normalize(K.gradients(loss, conv_output)[0])
    inp = input_model.layers[0].input
    output, grads_val = K.function([inp], [conv_output, grads])([image])
    output, grads_val = output[0,:], grads_val[0, :, :, :]
    #---3. channel average of gradient
    weights = np.mean(grads_val, axis=(0,1))
    #---4. multiply averaged channel weights to last output
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (72, 24), cv2.INTER_LINEAR) #Modify here depending on input shape
    cam = np.maximum(cam, 0)
    heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return heatmap

def show_heatmap(heatmap):
    proj = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)

    fig = plt.figure()
    ax = plt.subplot(projection=proj)
    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs=mticker.MultipleLocator(90), ylocs=mticker.MultipleLocator(45),
                 linestyle='-', color = 'gray')
    mat = ax.matshow(heatmap, cmap='BuPu', extent=img_extent, transform=proj)
    cbar = fig.colorbar(mat, ax=ax, orientation='horizontal')
    plt.show()

