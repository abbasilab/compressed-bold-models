#Importing library
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse


def Alexnet(weights_path):
    #Instantiation
    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(None,None,3), kernel_size=(11,11), strides=(4,4), padding='same',name="conv1"))
    AlexNet.add(Activation('relu',name="relu1"))
    AlexNet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same',name="pool1"))
    #AlexNet.add(BatchNormalization(name="norm11"))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same',name="conv2",groups=2))
    AlexNet.add(Activation('relu',name="relu2"))
    AlexNet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same',name="pool2"))
    #AlexNet.add(BatchNormalization(name="norm22"))


    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',name="conv3"))
    AlexNet.add(Activation('relu',name="relu3"))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',name="conv4",groups=2))
    AlexNet.add(Activation('relu',name="relu4"))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',name="conv5",groups=2))
    AlexNet.add(Activation('relu',name="relu5"))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same',name="pool5"))

    #Passing it to a Fully Connected layer
    # AlexNet.add(Flatten())
    # # 1st Fully Connected Layer
    # AlexNet.add(Dense(4096, input_shape=(None,None,3,),name="fc6"))
    # AlexNet.add(Activation('relu',name="relu6"))
    # # Add Dropout to prevent overfitting
    # AlexNet.add(Dropout(0.5,name="drpo6"))

    # # #2nd Fully Connected Layer
    # AlexNet.add(Dense(4096,name="fc7"))
    # AlexNet.add(Activation('relu',name="relu7"))
    # #Add Dropout
    # AlexNet.add(Dropout(0.5,name="drpo7"))

    # #3rd Fully Connected Layer
    # AlexNet.add(Dense(1000,name="fc8"))
    # AlexNet.add(Activation('relu',name="relu8"))

    # #Output Layer
    # AlexNet.add(Dense(10))
    # AlexNet.add(Activation('softmax',name="prob"))

    #Model Summary
    AlexNet.summary()

    AlexNet.load_weights(weights_path,by_name=True)
    
    return AlexNet


def visualize_filter(filter_index, feature_extractor):
    # We run gradient ascent for 20 steps
    iterations = 10
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def compute_loss(input_image, filter_index, feature_extractor):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


def initialize_image():
    img_width =227
    img_height = 227
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    return (img - 0.5) * 0.25



def filter_visualization(layer_name, top_filters, weights_path):
    # The dimensions of our input image
    # layer_name: str for example 'conv5'
    # top_fliter: list including the index of top filtrs    

    model = Alexnet(weights_path)
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    for i in top_filters:
        loss, img = visualize_filter(i,feature_extractor)
        keras.preprocessing.image.save_img(f"./{i}.jpg", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lay_name', type=str, required=True, help='layer index')
    parser.add_argument('--top _ilter', type=list, required=True, help='how many filters to remove')
    parser.add_argument('--weights_path', type=str, required=True, help='where UC maps are saved')
    args = parser.parse_args()   

    filter_visualization(args.lay_name, args.top_filter, args.weights_path)