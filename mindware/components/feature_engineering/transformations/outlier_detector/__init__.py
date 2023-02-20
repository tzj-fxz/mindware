import os
from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
outlier_detector_directory = os.path.split(__file__)[0]
_outlier_detector = find_components(__package__, outlier_detector_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_outlier_detector(detector):
    _addons.add_component(detector)
