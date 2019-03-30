from keras.models import load_model

classifierOne = load_model('model_without_transfer_learning.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/train/seborrheic_keratosis/ISIC_0012183.jpg', target_size = (128, 128, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifierOne.predict(test_image)
p = np.argmax(result)
if p==0: print("basal")
elif p==1: print("melanoma")
elif p==2: print("squamous")
print(p)

classifierTwo = load_model('model_using_transfer_learning.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/train/seborrheic_keratosis/ISIC_0012183.jpg', target_size = (229, 229))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifierTwo.predict(test_image)
p = np.argmax(result)
if p==0: print("basal")
elif p==1: print("melanoma")
elif p==2: print("squamous")
print(p)