from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from .AttModel import *

def setup(opt):
    model = AttModel(opt)
    if vars(opt).get('start_from', None) is not None:
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model