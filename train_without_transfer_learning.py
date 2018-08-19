from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model 

classifier = Sequential()
classifier.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.1))
classifier.add(GlobalAveragePooling2D())
classifier.add(Dense(3, activation='softmax'))
classifier.summary()
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory("dataset/train",target_size = (128, 128),batch_size = 50,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory("dataset/valid",target_size = (128, 128),batch_size = 50,class_mode = 'categorical')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("model_without_transfer_learning.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
classifier.fit_generator(training_set, epochs = 100, validation_data = test_set, callbacks = [checkpoint], shuffle=True)