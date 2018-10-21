from creating_cnn import *

flower_recognizer_model.load_weights('flower_vegetable_model.h5')

image_to_recognize = image.load_img('test/purple_rose.jpg', target_size=(100,100))
image_to_recognize = image.img_to_array(image_to_recognize)
image_to_recognize = np.expand_dims(image_to_recognize, axis = 0)
prediction = flower_recognizer_model.predict(image_to_recognize)

print(os.listdir('flowers/'))
print (prediction)