from ast import Raise
import types
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

def parse_extractor_str(model_str):
    """Function used to parse the extractor creator function from a string

    Args:
        model_str (str): the name of the model

    Returns:
        fn : The function that creates the model from tf.keras.applications
    """
    if model_str.lower() == "xception":
        return tf.keras.applications.Xception
    elif model_str.lower() == "densenet201":
        return tf.keras.applications.DenseNet201
    elif model_str.lower() == "efficientnetv2s":
        return tf.keras.applications.EfficientNetV2S
    elif model_str.lower() == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0
    elif model_str.lower() == "DenseNet121".lower():
        return tf.keras.applications.DenseNet121
    elif model_str.lower() == "EfficientNetB2".lower():
        return tf.keras.applications.EfficientNetB2
    elif model_str.lower() == "EfficientNetV2B1".lower():
        return tf.keras.applications.EfficientNetV2B1
    else:
        print(ValueError("Model not recogniezed"))
        Raise(ValueError("Model not recogniezed"))


def parse_classifier(type, num_classes, params):
    """Function to parse and create the classifier model. For the moment, only the dense type classifier is supported

    Args:
        type (architecture type): The type of architecture, for the moment only "dense" is supported
        num_classes (int): The number of classification classes
        params (list): model parameters

    Returns:
        list: List containing the resulting classifier architecture
    """
    if type == "dense":
        class_layers = []
        for i in range(len(params)):
            class_layers.append(layers.Dense(params[i], activation='relu', dtype= tf.float16))
        class_layers.append(layers.Dense(num_classes))
        class_layers.append(layers.Activation("softmax", dtype=tf.float32, name="softmax_float32"))
        return class_layers
    else :
        print("Head type not recognized")
        Raise(ValueError("Head type not recognized"))
        




class full_model():
    """Full model wrapper class that includes a feature extractor backbone and a classifier head
    """
    def __init__(self, 
                 input_shape = (224, 224, 3),
                 num_classes = 101 ,
                 extractor_name = "efficientnetv2s",
                 classifier_type = "dense", 
                 classifier_parameters = []):
        self.model = create_model(input_shape, num_classes  , extractor_name , classifier_type , classifier_parameters )
    
    def compile(self,unfreeze_extractor=False, *args, **kwargs):
        """Compiles the model with the ability of freezing/unfreezing the feature extractor

        Args:
            unfreeze_extractor (bool, optional): Whether to unfreee the feature extractor or not. Defaults to False.
        """
        if unfreeze_extractor:
            self.model.layers[1].trainable = True
        self.model.compile(*args, **kwargs)


def create_model(input_shape = (224, 224, 3),
                 num_classes = 101 ,
                 extractor_name = "efficientnetv2s",
                 classifier_type = "dense", 
                 classifier_parameters = []):
    """Creates a full model with a feature extractor backbone and a classifier head from parameters
    """
    # Getting the extractor
    extractor = parse_extractor_str(extractor_name)(include_top=False)
    extractor.trainable = False # freeze base model layers
    # Getting the classifier
    classifier_layers = parse_classifier(classifier_type, num_classes, classifier_parameters)
    
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = extractor(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    for layer in classifier_layers:
        x = layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
    
    



