import os
from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
mislabel_detector_directory = os.path.split(__file__)[0]
_mislabel_detector = find_components(__package__, mislabel_detector_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_mislabel_detector(detector):
    _addons.add_component(detector)
