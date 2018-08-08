from keras.models import load_model
classifier = load_model('model.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test/.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)