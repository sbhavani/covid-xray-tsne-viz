import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorboard.plugins import projector
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import pickle


## Globals
PATH = os.getcwd()
IMGPATH  = os.path.join(PATH, 'images')
## Path to save the embedding and checkpoints generated
LOG_DIR = PATH + '/logs/log-vggnet/'
# os.mkdir(LOG_DIR)

METADATA_FNAME = 'metadata_4_classes.tsv'

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def collect_metadata(df):
    num_of_samples=len(df)
    num_of_samples_each_class = None
    names = df['finding'].unique()
    print(names)
    y = df['finding'].astype('category').cat.codes.tolist()
    metadata_file = open(os.path.join(LOG_DIR, METADATA_FNAME), 'w')
    metadata_file.write('Class\tName\n')
    j=0
    c_prev = None
    for i in range(num_of_samples):
        c = names[y[i]]
        if i==0:
            c_prev = c        
        # change in class
        if c != c_prev:
            c_prev=c
            j+=1
        metadata_file.write('{}\t{}\n'.format(j,c))
    metadata_file.close()


def generate_feat_vecs(fnames):
    model = VGG16(weights='imagenet', include_top=False)
    img_data=[]
    imgvecs = []
    cnt = 0
    for i in fnames:
        print('Processing ' + i)
        # read and prepare image
        x = image.load_img(os.path.join(IMGPATH, i), target_size=(224, 224))
        im = image.img_to_array(x)
        img_data.append(im)
        im = np.expand_dims(im, 0)
        im = preprocess_input(im)
        y = model.predict(im).flatten()
        imgvecs.append(y)
        cnt += 1

    img_data = np.array(img_data)
    img_data.dump('img_data.pkl')
    feature_vectors = np.stack(imgvecs, axis=0)
    feature_vectors.dump('featvec.pkl')

    print ("feature_vectors_shape:",feature_vectors.shape)
    print ("size of individual feature vector:",feature_vectors.shape[1])

    return img_data, feature_vectors


df = pd.read_csv('metadata.csv')

# filter out CT scans
df_cxr = df[df['view'] != 'CT']

# sort by indication
df_sorted = df_cxr.sort_values(['finding'])
print(df_sorted.head())

## collect metadata
collect_metadata(df_sorted)

## get images and feat vecs
if os.path.exists(os.path.join(PATH, 'featvec.pkl')):
    img_data = pickle.load(open('img_data.pkl', 'rb'))
    feature_vectors = pickle.load(open('featvec.pkl', 'rb'))
else:
    img_data, feature_vectors = generate_feat_vecs(df_sorted['filename'])

sprite = images_to_sprite(img_data)
imsave(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)

## visualize features
features = tf.Variable(feature_vectors, name='features')
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, METADATA_FNAME)
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
