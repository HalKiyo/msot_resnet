import numpy as np
import numpy.ma as ma

def load(inkey, outkey):
    infile = f"/docker/mnt/d/research/D2/resnet/predictors/{inkey}.npy"
    outfile = f"/docker/mnt/d/research/D2/resnet/predictant/class/{outkey}.npy"
    predictors = np.load(infile)
    predictant = np.load(outfile)
    return predictors, predictant

def mask(x):
    m = ma.masked_where(x>9999, x)
    z = ma.masked_where(m==0, m)
    f = ma.filled(z, 0)
    return f

def shuffle(indata, outdata, vsample, seed):
    rng = np.random.default_rng(seed)

    outdata = outdata.reshape(42, 165)
    random_number = indata.shape[1]*indata.shape[2]
    random_index = rng.choice(random_number, random_number, replace=False)

    train_index = random_index[:-vsample]
    train_dct = {'model': train_index//indata.shape[2],
                 'year': train_index%indata.shape[2]}
    x_train = np.array([ indata[:, m, y] for m, y in zip(
                         train_dct['model'], train_dct['year']) ])
    y_train = np.array([ outdata[m, y] for m, y in zip(
                         train_dct['model'], train_dct['year']) ])

    val_index = random_index[-vsample:]
    val_dct = {'model': val_index//indata.shape[2],
                 'year': val_index%indata.shape[2]}
    x_val = np.array([ indata[:, m, y] for m, y in zip(
                         val_dct['model'], train_dct['year']) ])
    y_val = np.array([ outdata[m, y] for m, y in zip(
                         val_dct['model'], train_dct['year']) ])
    return x_train, y_train, x_val, y_val, train_dct, val_dct

