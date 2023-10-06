# -*- coding: utf-8 -*-
# @Author: Pengyao Ping
# @Date:   2023-01-24 15:47:23
# @Last Modified by:   Pengyao Ping
# @Last Modified time: 2023-09-07 14:03:54

__version__ = "0.2.3"

import pkg_resources

weights_i = pkg_resources.resource_filename('deepdirect', '/weights/model_i_weights.h5')
weights_d = pkg_resources.resource_filename('deepdirect', '/weights/model_d_weights.h5')
