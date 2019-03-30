from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model 

from tensorflow.keras import applications
pre_model = applications.InceptionV3(weights = "imagenet", include_top=False)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
path = "dataset/train"
path_test = "dataset/test"
training_set = train_datagen.flow_from_directory(path,target_size = (299, 299),batch_size = 8, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(path_test,target_size = (299, 299),batch_size = 8, class_mode = 'categorical')

for layer in pre_model.layers[:5]:
    layer.trainable = False

x = pre_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation="softmax")(x)
model = Model(inputs=pre_model.input, outputs=predictions)

model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("model_using_transfer_learning.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit_generator(training_set, epochs = 10, validation_data = test_set, callbacks = [checkpoint], shuffle=True)