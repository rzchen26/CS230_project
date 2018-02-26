from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import sunpy.map
import tensorflow as tf
import os
import sunpy.map

# load in fits file
AIA=fits.open("raw_data/AIA_0h_compiled.fits")
Mag=fits.open("raw_data/Mag_0h_compiled.fits")
size_raw_data=AIA[0].data.shape[0]

#ration to split test and dev set
ratio_test=0.1
ratio_dev=0.1

# save data to train dev test folders respectively
np.random.seed(0)
for eg in range(size_raw_data):
    dice=np.random.rand()
    if dice < ratio_test:
        filedir="data/test/"
    elif dice < (ratio_test+ratio_dev):
        filedir="data/dev/"
    else:
        filedir="data/train/"
    X=np.log(np.abs(AIA[0].data[eg,:,:])+1)/10
    Y=np.log(np.abs(Mag[0].data[eg,:,:])+1)/10

    outfile=filedir+"pair_"+str(eg)+".tfrecord"
    writer = tf.python_io.TFRecordWriter(outfile)

    feature = {}
    feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
    feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=Y.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=feature))

         # Serialize the example to a string
    serialized = example.SerializeToString()

         # write the serialized objec to the disk
    writer.write(serialized)
    writer.close()
