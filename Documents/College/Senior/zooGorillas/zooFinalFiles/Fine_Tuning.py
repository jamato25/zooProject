from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions 
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense

# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'fineTuning_notNorm.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'GorillaSquareCrop/train'
validation_data_dir = 'GorillaSquareCrop/validate'
nb_train_samples = 600
nb_validation_samples = 90
epochs = 20
batch_size = 5

amani_train = 100
amani_validate=15

honi_train=100
honi_validate=15

kira_train=100
kira_validate=15

kuchimba_train=100
kuchimba_validate=15

louis_train=100
louis_validate=15

motuba_train=100
motuba_validate=15


base_model = applications.VGG16(weights='imagenet',include_top= False,input_shape=(img_height,img_width,3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(6, activation='softmax'))
model = Model(input= base_model.input, output= top_model(base_model.output))



# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              optimizer='rmsprop',
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save(top_model_weights_path)
