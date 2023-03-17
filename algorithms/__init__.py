from .adaprox import adaprox
from .universal_mirror_prox import universal_mirror_prox
from .extra_gradient import extra_gradient

from .projections import *

algorithms = {
    'Extra Gradient': extra_gradient,
    'Universal Mirror Prox': universal_mirror_prox,
    'AdaProx': adaprox,
}
