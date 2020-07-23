import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

#The model file will be named whatever top model weight path is
top_model_weights_path = 'Bottleneck_Norm_150_augment.h5'
#Change train_data_dir and validation_data_dir to whatever directory you have the 
#images contained in
train_data_dir = 'GorillaSquareCropNormalized/train'
validation_data_dir = 'GorillaSquareCropNormalized/validate'
nb_train_samples = 600
nb_validation_samples = 90
epochs = 40
#Number of train samples and number of validation samples must be divisible by batch size
batch_size = 5

amani_train =100
amani_validate = 15

honi_train = 100
honi_validate = 15

kira_train = 100
kira_validate = 15

kuchimba_train = 100
kuchimba_validate = 15

louis_train = 100
louis_validate = 15

motuba_train = 100
motuba_validate = 15

def save_bottlebeck_features():
    #augmentation configuration for model
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples//batch_size)

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples//batch_size)

    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))

#Make sure to change this array for whatever the number of classes is
    train_labels = np.array(
        [0] * (amani_train) + [1] * (honi_train) + [2] * (kira_train) + [3] * (kuchimba_train)
        + [4] * (louis_train) + [5] * (motuba_train))

    print('TRAIN LABEL SHAPE:', train_labels.shape)
    print('TRAIN LABELS ARRAY:', train_labels)
    train_labels.shape[1:]
    print('TRAIN LABEL SHAPE AFTER RESHAPING:', train_labels.shape)
    print('TRAIN LABELS ARRAY AFTER RESHAPING:', train_labels)
    validation_data = np.load(open('bottleneck_features_validation.npy'))

#Make sure to change this array for whatever the number of classes is
    validation_labels = np.array(
        [0] * (amani_validate) + [1] * (honi_validate) + [2] * (kira_validate) + [3] * (kuchimba_validate)
        + [4] * (louis_validate) + [5] * (motuba_validate))

    print('TRAIN DATA SHAPE:', train_data.shape)
    print('VALIDATION DATA SHAPE:', validation_data.shape)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #Dense number must be equal to the number of classes
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    print('TRAIN LABELS', train_labels.shape)
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()