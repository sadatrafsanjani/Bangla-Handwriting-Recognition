import numpy as np
import glob as gb
import cv2
import random
import matplotlib.pyplot as plt
import keras.backend as K
import itertools
from imgaug import augmenters as iaa
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate


trainings = []
labels = []
DIM = 60
np.random.seed(1337)
batch_size = 128
nb_classes = 84
nb_epoch = 12
class_names = []



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    Plotting the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def listShuffling(trainings, labels):
    """
    Shuffling the training and label lists 
    """
    c = list(zip(trainings, labels))
    random.shuffle(c)
    a, b = zip(*c)
    
    return a, b


def readImage(picture):
    """
    Reading each image with opencv 
    """    
    return cv2.imread(picture, 0)


def viewImage(image):
    """
    View Image 
    """
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def formulation(image, directory):
    
    container = np.array([[]], dtype='float32')
    
    container = image.ravel() # Reshaping image in 3600
    t = container.astype(np.float32) / 255.0
    trainings.append(t) # Creating image training list
    labels.append(directory) # Creating image label list
    
    


def prepareTrainingSet():
    """
    Preparing the image training list and image label list 
    """
    
    r = random.randint(1,6)

    seq = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-r, r), "y": (-r, r)}),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Affine(scale={"x": (r, r), "y": (r, r)}),
            iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255))
            ])

    
    for directory in range(0, 84):
        print("---------------------", directory)
        class_names.append(directory)
        for picture in gb.glob("./cluttered/"+str(directory)+"/*.png"):

            img = readImage(picture) # Reading image of 60 x 60 pixels
            formulation(img, directory)
            
            ## Augmentation
            img = img[:, :, np.newaxis] # Adding image channel 1
            img_aug = seq.augment_image(img) # Applying augmentation
            img_transposed = img_aug.transpose(2,0,1) 
            img_reshaped = img_transposed.reshape(-1, img_aug.shape[1]) # Reshaping in 60 x 60
            formulation(img_reshaped, directory)
            
    
    return listShuffling(trainings, labels); # Shuffling lists

    
# Initializing image training list and image label list 
X, Y = prepareTrainingSet()
class_names = np.array(class_names)
trainings = []
labels = []

# Data Split
samples = len(Y)

train_samples = int(samples * 0.8)
X_train = np.array(X[ : train_samples])
y_train = np.asarray(Y[ : train_samples], dtype=np.int64)

valid_samples = int(samples * 0.05)
X_valid = np.array(X[train_samples : train_samples + valid_samples])
y_valid = np.asarray(Y[train_samples : train_samples + valid_samples], dtype=np.int64)

test_samples = int(samples * 0.15)
X_test = np.array(X[-test_samples : ])
y_test = np.asarray(Y[-test_samples : ], dtype=np.int64)


X = []
Y = []


# Reshape for convolutions
X_train = X_train.reshape((X_train.shape[0], DIM, DIM, 1))
X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, 1))
X_test = X_test.reshape((X_test.shape[0], DIM, DIM, 1))


y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

input_shape = (60, 60, 1)

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]


#Localization

visible = Input(shape=input_shape)

#Convolution
conv1 = Convolution2D(32, kernel_size=(5, 5), activation='relu', padding='same')(visible)
conv2 = Convolution2D(32, kernel_size=(5, 5), activation='relu', padding='same')(conv1)
#Pooling
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

#Dropout
dropout1 = Dropout(.25)(pool1)

#Convolution
conv6 = Convolution2D(64, kernel_size=(5, 5), activation='relu', padding='same')(dropout1)
conv7 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

conv8 = Convolution2D(64, kernel_size=(5, 5), activation='relu', padding='same')(dropout1)
conv9 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv8)

#Merge
merge = concatenate([conv7, conv9])

#Pooling
pool2 = MaxPooling2D(pool_size=(2, 2))(merge)

#Dropout
dropout2 = Dropout(.25)(pool2)

#Flatten
flatten = Flatten()(dropout2)

#Dense
fully = Dense(units=1280, activation='relu')(flatten)

#Final
output = Dense(units=nb_classes, activation='softmax')(fully)

#Training
model = Model(inputs=visible, outputs=output)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

XX = model.input
YY = model.layers[0].output
F = K.function([XX], [YY])

nb_epochs = 100
batch_size = 86
fig = plt.figure()


checkpoint = ModelCheckpoint('borno.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit( X_train, y_train, validation_data = (X_test, y_test), epochs = nb_epochs, batch_size = batch_size, callbacks=[checkpoint] )

#model.load_weights('borno.h5')
#model.fit( X_train, y_train, validation_data = (X_test, y_test), epochs = nb_epochs, batch_size = batch_size)


# Plotting Confusion Matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Non-Normalized')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized')
plt.show()
