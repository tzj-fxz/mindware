import os
from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
imputer_directory = os.path.split(__file__)[0]
_imputer_cat = find_components(__package__, imputer_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_imputer_cat(imputer):
    _addons.add_component(imputer)
