from os.path import exists
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from preprocess import load, shuffle, mask
from model3 import build_model
from gradcam import image_preprocess, grad_cam, show_heatmap, average_heatmap

disable_eager_execution()

def main():
    #---1. dataset
    save_flag = False#MODIFALABLE
    vsample = 1000#MODIFALABLE
    tors = 'predictors_coarse_std_Apr_t'
    tant = 'pr_1x1_std_MJJASO_one'#MODIFALABLE
    savefile = f"/docker/mnt/d/research/D2/cnn3/train_val/{tors}-{tant}.pickle"
    if exists(savefile) is True and save_flag is False:
        with open(savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
    else:
        predictors, predictant = load(tors, tant)
        x_train, y_train, x_val, y_val, train_dct, val_dct = shuffle(predictors, predictant, vsample)
        x_train, x_val = mask(x_train), mask(x_val)
        x_train, x_val = x_train.transpose(0,2,3,1), x_val.transpose(0,2,3,1)

    #---2, training
    val_nm = 1#MODIFALABLE
    lat, lon = 24, 72#MODIFALABLE
    batch_size = 256#MODIFALABLE
    epochs = 100#MODIFALABLE
    lr = 0.0001#MODIFALABLE
    model = build_model((lat, lon, val_nm))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    weights_path = f"/docker/mnt/d/research/D2/cnn3/weights/{tors}-{tant}.h5"
    if exists(weights_path) is True and save_flag is False:
        model.load_weights(weights_path)
    else:
        his = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        #model.summary()

    #---3, validation
    pred = model.predict(x_val)[:,0]
    corr = np.corrcoef(pred, y_val)[0,1]
    plt.scatter(pred, y_val, color='pink')
    plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), color='green')
    plt.title(f"{corr}")
    plt.show()

    #---4. gradcam
    index = 700#MODIFALABLE
    layer_name = 'conv2d_2'#MODIFALABLE
    preprocessed_image = image_preprocess(x_val, index)
    heatmap = grad_cam(model, preprocessed_image, y_val, layer_name, lat, lon)
    #show_heatmap(heatmap)
    average_heatmap(x_val, model, y_val, layer_name, lat, lon, num=300)

    #---5. save environment
    if save_flag is True:
        model.save_weights(weights_path)
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(savefile, 'wb') as f:
            pickle.dump(dct, f)
        print(f"{savefile} and weights are saved")
    else:
        print(f"save_flag is {save_flag} not saved")

if __name__ == '__main__':
    main()

