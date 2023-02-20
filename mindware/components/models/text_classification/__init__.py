import os
from mindware.components.models.base_nn import BaseTextClassificationNeuralNetwork
from mindware.components.utils.class_loader import ThirdPartyComponents, find_components

"""
Load the buildin classfier
"""
text_classifiers_directory = os.path.split(__file__)[0]
_text_classifiers = find_components(__package__, text_classifiers_directory, BaseTextClassificationNeuralNetwork)


"""
Load third-party classifier
"""
_addons = ThirdPartyComponents(BaseTextClassificationNeuralNetwork)

def add_classifier(classifier):
    _addons.add_component(classifier)