from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["Conv2dOp"],onnx_op=["Conv"])
class Conv:
    @classmethod
    def version_1(cls,ctx,node,**kwargs):
        kernel_shape = ctx._shapes[node._inputs[1]][2:]
        pads = [node.get_attr_value('padding',0)]*4
        strides = [node.get_attr_value('stride',1)]*2
        node.set_attr('kernel_shape',kernel_shape)
        node.set_attr('pads',pads)
        node.set_attr('strides',strides)


    @classmethod
    def version_11(cls,ctx,node,**kwargs):

        cls.version_1(ctx,node,**kwargs)