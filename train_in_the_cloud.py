from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
path = "dataset/train"
path_test = "dataset/test"
training_set = train_datagen.flow_from_directory(path,target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(path_test,target_size = (64, 64),batch_size = 32,class_mode = 'categorical')

from keras import applications

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (64, 64, 3))
for layer in model.layers[:5]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(100, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])

# Save the model according to the conditions  
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model_final.fit_generator(training_set,steps_per_epoch = 10,epochs = 10,validation_data = test_set,validation_steps = 2,
callbacks = [checkpoint])