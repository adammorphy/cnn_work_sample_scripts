from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model
import keras.backend as K
from tensorflow import pad, constant, keras, random
import os
import tensorflow_addons as tfa

# Modified example work script from implimenting Convolutional Neural Networks as a feature extractor in TensorFlow.
# This script is intended as a work example of Adam Morphy's work with the Vancouver Whitecaps FC, and has been modified outside its original data pipeline.

#################################################
#                   _______
#                  |       |              
#            o     |       |
#          -()-   o
#           |\
#################################################

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ConvNet:
    """
    Convolutional neural network architecture

    ...

    Attributes
    ----------
    pass_input_channels : keras.layers.Input
        input layer to CNN representing pass channels
    destination_input : keras.layers.Input
        input layer to CNN representing the target/destination of pass
    model_name : str
        name of the model, one of: completion, value, selection
    combined: tensorflow.Tensor
        surface prediction and target destination channel with shape (None, length, width, 2)
    pixel: float
        models prediction value at target location

    Methods
    -------
    get_surface_model():
        Returns a surface model object
    get_prediction_model():
        Returns a prediction model object

    """

    def __init__(self, params, model_name):
        """
        Constructs all the necessary attributes and model object for the CNN.

        Parameters
        ----------
            params : dict
                contains architecture parameters for model
            model_name : str
                name of the model, one of: completion, value, selection
        """
        self.pass_input_channels = Input(
            shape=(params["x_bins"], params["y_bins"], params["number_input_channels"]),
            name="pass_input_channels",
        )
        self.destination_input = Input(
            shape=(params["x_bins"], params["y_bins"], 1), name="destination_input"
        )

        self.model_name = model_name

        padding_5 = constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        padding_3 = constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        random.set_seed(1)

        # 2 Conv Layers
        x = Conv2D(32, (5, 5), strides=(1, 1), activation="relu", padding="valid")(
            self.pass_input_channels
        )
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")
        x = Conv2D(64, (5, 5), strides=(1, 1), activation="relu", padding="valid")(x)
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")

        # PREDICTION at 1x
        pred1 = Conv2D(32, (1, 1), activation="relu", padding="valid")(x)
        pred1 = Conv2D(1, (1, 1), activation="linear")(pred1)

        # DOWNSAMPLE 1
        x = MaxPooling2D((2, 2), padding="valid")(pred1)
        # 2 Conv Layers
        x = Conv2D(32, (5, 5), strides=(1, 1), activation="relu", padding="valid")(x)
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")
        x = Conv2D(64, (5, 5), strides=(1, 1), activation="relu", padding="valid")(x)
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")

        # PREDICTION at 0.5x
        pred2 = Conv2D(32, (1, 1), activation="relu", padding="valid")(x)
        pred2 = Conv2D(1, (1, 1), activation="linear")(pred2)

        # DOWNSAMPLE 2
        x = MaxPooling2D((2, 2), padding="valid")(pred2)
        # 2 Conv Layers
        x = Conv2D(32, (5, 5), strides=(1, 1), activation="relu", padding="valid")(x)
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")
        x = Conv2D(64, (5, 5), strides=(1, 1), activation="relu", padding="valid")(x)
        x = pad(tensor=x, paddings=padding_5, mode="SYMMETRIC")

        # PREDICTION at 0.25x
        pred3 = Conv2D(32, (1, 1), strides=(1, 1), activation="relu", padding="valid")(
            x
        )
        pred3 = Conv2D(1, (1, 1), activation="linear")(pred3)

        # Upsample Prediction 3 from 0.25x to 0.5x
        pred3 = UpSampling2D((2, 2))(pred3)
        pred3 = Conv2D(32, (3, 3), activation="relu", padding="valid")(pred3)
        pred3 = pad(tensor=pred3, paddings=padding_3, mode="SYMMETRIC")
        pred3 = Conv2D(1, (3, 3), activation="linear", padding="valid")(pred3)
        pred3 = pad(tensor=pred3, paddings=padding_3, mode="SYMMETRIC")

        # CONCATINATION pred2 and pred3- FUSION LAYER 1
        combined2and3 = Concatenate()([pred2, pred3])
        combined2and3 = Conv2D(1, (1, 1), activation="linear")(combined2and3)

        # Upsample combined2and3 predictions from 0.50x to 1x
        combined2and3 = UpSampling2D((2, 2))(combined2and3)
        combined2and3 = Conv2D(32, (3, 3), activation="relu", padding="valid")(
            combined2and3
        )
        combined2and3 = pad(tensor=combined2and3, paddings=padding_3, mode="SYMMETRIC")
        combined2and3 = Conv2D(1, (3, 3), activation="linear", padding="valid")(
            combined2and3
        )
        combined2and3 = pad(tensor=combined2and3, paddings=padding_3, mode="SYMMETRIC")

        # CONCATINATION combined2and3 and pred1 - FUSION LAYER 2
        combined1and2and3 = Concatenate()([pred1, combined2and3])
        combined1and2and3 = Conv2D(1, (1, 1), activation="linear")(combined1and2and3)

        # FINAL PREDICTION SURFACE
        out = Conv2D(32, (1, 1), strides=(1, 1), activation="relu", padding="valid")(
            combined1and2and3
        )
        out = Conv2D(1, (1, 1), activation="linear")(out)

        # FINAL ACTIVATION
        if self.model_name == "completion":
            out = keras.activations.sigmoid(out)

            self.combined = Concatenate()([out, self.destination_input])

            surface = self.combined[:, :, :, 0]
            mask = self.combined[:, :, :, 1]
            masked = surface * mask
            self.pixel = K.sum(masked, axis=(2, 1))

        elif self.model_name == "selection":
            out = keras.activations.softmax(out, axis=(1, 2))

            self.combined = Concatenate()([out, self.destination_input])
            surface = self.combined[:, :, :, 0]
            mask = self.combined[:, :, :, 1]
            masked = surface * mask
            self.pixel = K.sum(masked, axis=(2, 1))

        elif self.model_name == "value":
            out = keras.activations.sigmoid(out)
            out = tfa.layers.InstanceNormalization()(out)

            self.combined = Concatenate()([out, self.destination_input])

            surface = self.combined[:, :, :, 0]
            mask = self.combined[:, :, :, 1]
            masked = surface * mask
            self.pixel = K.sum(masked, axis=(2, 1))

        else:
            raise (
                ValueError(
                    f"{self.model_name} not a recognized model type. Choices are: completion, selection, and value"
                )
            )

    def get_surface_model(self):
        """
        Returns surface model object

        Parameters
        ----------
        None

        Returns
        -------
        tf.keras.Model object, surface model
        """
        surface_model = Model(
            [self.pass_input_channels, self.destination_input], self.combined
        )
        surface_model._name = self.model_name
        return surface_model

    def get_prediction_model(self):
        """
        Returns prediction model object

        Parameters
        ----------
        None

        Returns
        -------
        tf.keras.Model object, prediction model
        """
        prediction_model = Model(
            [self.pass_input_channels, self.destination_input],
            self.pixel,
        )
        prediction_model._name = self.model_name
        return prediction_model
