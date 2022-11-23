import tensorflow as tf
from tensorflow import keras
import random
import numpy as np


def save_models(bestModel, secondBestModel):

    model_json = bestModel.to_json()
    with open("models/bestModel.json", "w") as json_file:
        json_file.write(model_json)
    bestModel.save_weights("models/bestModel.h5")
    model_json_2 = secondBestModel.to_json()
    with open("models/secondModel.json", "w") as json_file:
        json_file.write(model_json_2)
    secondBestModel.save_weights("models/secondModel.h5")


def load_models():

    json_file = open('models/bestModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    bestModel = tf.keras.models.model_from_json(loaded_model_json)
    bestModel.load_weights("models/bestModel.h5")

    json_file = open('models/secondModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    secondBestModel = tf.keras.models.model_from_json(loaded_model_json)
    secondBestModel.load_weights("models/secondModel.h5")

    return bestModel, secondBestModel


def generate_initial_models():

    # 4 parameters to pass in: velocity, flappy bird x and y values, current pipe y-value
    initializer = keras.initializers.RandomNormal(mean=0., stddev=1.)

    l1 = keras.layers.Dense(16, kernel_initializer=initializer, bias_initializer=initializer, input_shape=(4,))
    l2 = keras.layers.Dense(16, kernel_initializer=initializer, bias_initializer=initializer)
    out = keras.layers.Dense(2, kernel_initializer=initializer, bias_initializer=initializer, activation='softmax')

    model = keras.Sequential(layers=[l1, l2, out])

    model.build()

    l1 = keras.layers.Dense(16, kernel_initializer=initializer, bias_initializer=initializer, input_shape=(4,))
    l2 = keras.layers.Dense(16, kernel_initializer=initializer, bias_initializer=initializer)
    out = keras.layers.Dense(2, kernel_initializer=initializer, bias_initializer=initializer, activation='softmax')

    model2 = keras.Sequential(layers=[l1, l2, out])
    model2.build()

    save_models(model, model2)


def create_random_variation(model1, model2):
    newModel = keras.Sequential()
    for idx in range(len(model1.layers)):
        curLayer = None
        if random.randint(0, 1) == 0:
            curLayer = model1.layers[idx].get_weights()
        else:
            curLayer = model2.layers[idx].get_weights()
        numWeightsChange = random.randint(5, 20)
        while numWeightsChange > 0:
            if random.randint(0, 1) == 0:
                curPlace = random.randint(0, len(curLayer[0])-1)
                lengthOf = len(curLayer[0][curPlace]) - 1
                curLayer[0][curPlace][lengthOf] = random.randint(-8, 8)
            else:
                curLayer[1][random.randint(0, 1)] = random.randint(-8, 8)
            numWeightsChange -= 1
        if idx < 2:
            l1 = keras.layers.Dense(16, input_shape=(4,))
            newModel.add(l1)
            newModel.layers[idx].set_weights(curLayer)
        else:
            l1 = keras.layers.Dense(2, activation='softmax')
            newModel.add(l1)
            newModel.layers[idx].set_weights(curLayer)
    return newModel


def predict_for_model(model, velocity, bird_x, bird_y, pipe_y):
    input_tensor = np.array([velocity, bird_x, bird_y, pipe_y])
    input_tensor = tf.reshape(input_tensor, shape=(1, 4))
    return model.predict(input_tensor)

# generate_initial_models()
#
# print(predict_for_model(create_random_variation(load_models()[0], load_models()[1]), 0, -1, 1, 1))
