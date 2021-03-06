
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

get_ipython().magic(u'matplotlib inline')


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[2]:

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(filename):
                filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
                print('Found and verified', filename)
        else:
                raise Exception(
                        'Failed to verify' + filename + '. Can you get to it with a browser?')
        return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

# In[4]:

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]    # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders
    
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# ---
# Problem 1
# ---------
# 
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
# 
# ---

# In[5]:

Image('notMNIST_large/A/Z29yaWxsYXogMi50dGY=.png', height=200, width=200)


# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# 
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# 
# A few images might not be readable, we'll just skip them.

# In[6]:

image_size = 28    # Pixel width and height.
pixel_depth = 255.0    # Number of levels per pixel.

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                                                 dtype=np.float32)
    image_index = 0
    print(folder)
    for image in os.listdir(folder):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - 
                                        pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                                        (num_images, min_num_images))
        
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
                
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    
    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# ---
# Problem 2
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# 
# ---

# In[7]:

with open('notMNIST_large/C.pickle', 'rb') as f:
        letter_set = pickle.load(f)

plt.imshow(letter_set[49], cmap="hot")


# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
# 
# ---

# In[8]:

tot = 0
lengths = []
for train in train_datasets:
    with open(train, 'rb') as f:
        letter_set = pickle.load(f)
        dsize = len(letter_set)
        tot += dsize
        lengths.append(dsize)

for l in lengths:
    print(float(l) / tot * 100)


# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# 
# Also create a validation dataset for hyperparameter tuning.

# In[9]:

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
        
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):             
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                                        
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
        
    return valid_dataset, valid_labels, train_dataset, train_labels
                        
                        
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# In[10]:

train_labels


# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[11]:

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# 
# ---

# In[12]:

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Finally, let's save the data for later reuse:

# In[13]:

pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[14]:

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

# In[15]:

def img_hash(img):
    difference = []
    width, height = img.shape

    for row in range(height):
        for col in range(width):
            if col != width-1:
                difference.append(img[col][row] > img[col+1][row])

    decimal_value = 0
    hex_string = []

    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0

    return ''.join(hex_string)


# In[23]:

train_hashes = []
for train_img in train_dataset:
    train_hashes.append(img_hash(train_img))

valid_hashes = []
for valid_img in valid_dataset:
    valid_hashes.append(img_hash(valid_img))

test_hashes = []
for test_img in test_dataset:
    test_hashes.append(img_hash(test_img))


# In[24]:

pickle_file = 'notMNIST_hash.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_hashes': train_hashes,
        'valid_hashes': valid_hashes,
        'test_hashes': test_hashes,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[25]:

with open('notMNIST_hash.pickle', 'rb') as f:
    hashes = pickle.load(f)


# In[77]:

from collections import defaultdict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)

def get_dup_indexes(seq):
    dups = []
    for dup in sorted(list_duplicates(seq)):
        dups.extend(dup[1][1:])
    return dups

train_dups = get_dup_indexes(hashes['train_hashes'])
test_dups = get_dup_indexes(hashes['test_hashes'])
valid_dups = get_dup_indexes(hashes['valid_hashes'])


# In[80]:

train_dataset_unique = np.delete(train_dataset, np.array(train_dups), axis=0)
train_labels_unique = np.delete(train_labels, np.array(train_dups), axis=0)
test_dataset_unique = np.delete(test_dataset, np.array(test_dups), axis=0)
test_labels_unique = np.delete(test_labels, np.array(test_dups), axis=0)
valid_dataset_unique = np.delete(valid_dataset, np.array(valid_dups), axis=0)
valid_labels_unique = np.delete(valid_labels, np.array(valid_dups), axis=0)

pickle_file = 'notMNIST_sanitised.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset_unique': train_dataset_unique,
        'train_labels_unique': train_labels_unique,
        'test_dataset_unique': test_dataset_unique,
        'test_labels_unique': test_labels_unique,
        'valid_dataset_unique': valid_dataset_unique,
        'valid_labels_unique': valid_labels_unique,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[81]:

print(train_dataset_unique.shape)
print(train_labels_unique.shape)
print(test_dataset_unique.shape)
print(test_labels_unique.shape)
print(valid_dataset_unique.shape)
print(valid_labels_unique.shape)


# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---

# In[82]:

with open('notMNIST.pickle', 'rb') as f:
    data = pickle.load(f)


# In[83]:

num_samples = 5000

X = data['train_dataset'][:num_samples]
X = np.reshape(X, (num_samples, 28*28))
y = data['train_labels'][:num_samples]

Xv = data['valid_dataset']
Xv = np.reshape(Xv, (len(Xv), 28*28))
yv = data['valid_labels']

Xt = data['test_dataset']
Xt = np.reshape(Xt, (len(Xt), 28*28))
yt = data['test_labels']


# In[84]:

for ns in [50, 100, 1000, 5000, 20000]:
    print('Num samples: {}'.format(ns))
    X = data['train_dataset'][:ns]
    X = np.reshape(X, (ns, 28*28))
    y = data['train_labels'][:ns]

    model = LogisticRegression(n_jobs=-1)
    fit = model.fit(X, y)
    score_v = model.score(Xv, yv)
    print('Validation score: {}'.format(score_v))
    score_t = model.score(Xt, yt)
    print('Test score: {}'.format(score_t))
    print('')


# In[85]:

with open('notMNIST_sanitised.pickle', 'rb') as f:
    data_unique = pickle.load(f)


# In[89]:

Xu = data_unique['train_dataset_unique'][:num_samples]
Xu = np.reshape(Xu, (num_samples, 28*28))
yu = data_unique['train_labels_unique'][:num_samples]

Xvu = data_unique['valid_dataset_unique']
Xvu = np.reshape(Xvu, (len(Xvu), 28*28))
yvu = data_unique['valid_labels_unique']

Xtu = data_unique['test_dataset_unique']
Xtu = np.reshape(Xtu, (len(Xtu), 28*28))
ytu = data_unique['test_labels_unique']


# In[90]:

for ns in [50, 100, 1000, 5000, 20000]:
    print('Num samples: {}'.format(ns))
    Xu = data['train_dataset'][:ns]
    Xu = np.reshape(Xu, (ns, 28*28))
    yu = data['train_labels'][:ns]

    model = LogisticRegression(n_jobs=8)
    fit = model.fit(Xu, yu)
    score_v = model.score(Xvu, yvu)
    print('Validation score: {}'.format(score_v))
    score_t = model.score(Xtu, ytu)
    print('Test score: {}'.format(score_t))
    print('')


# In[ ]:



