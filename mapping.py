
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'D:/UTA/Machine Learning/visual_question_answering/COCO-2015/val2014/COCO_val2014_000000000073.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

# Define the mapping function
def mapping_function(pixel_value):
    # Example mapping: invert pixel values
    return 255 - pixel_value

# Apply the mapping function to each pixel
mapped_image = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        mapped_image[i, j] = mapping_function(image[i, j])

# Display the original and mapped images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(mapped_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Mapped Image')
axs[1].axis('off')

plt.show()

'''


#import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = 'D:/UTA/Machine Learning/visual_question_answering/models/best_model.pt'
model = keras.models.load_model(model_path)

# Load the image
image_path = 'D:/UTA/Machine Learning/visual_question_answering/COCO-2015/val2014/COCO_val2014_000000000073.jpg'  # Replace with your image file path
image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

# Preprocess the image for the model
x = keras.preprocessing.image.img_to_array(image)
x = keras.applications.resnet_v2.preprocess_input(x)

# Predict the class probabilities and attention map
preds, attention_map = model.predict(tf.expand_dims(x, axis=0))

# Get the class label with the highest probability
class_index = tf.argmax(preds, axis=1)[0]
class_label = keras.applications.resnet_v2.decode_predictions(preds, top=1)[0][0][1]

# Reshape and normalize the attention map
attention_map = tf.reshape(attention_map, (7, 7))
attention_map = keras.preprocessing.image.smart_resize(attention_map.numpy(), (224, 224), interpolation='bilinear')
attention_map /= tf.reduce_max(attention_map)

# Display the original image and attention map
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[0].set_title(f'Class: {class_label}')

axs[1].imshow(attention_map, cmap='jet')
axs[1].set_title('Attention Map')

plt.show()
'''