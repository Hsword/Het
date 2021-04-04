from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["ConcatOp"],onnx_op=["Concat"])
class Concat:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

        # todo:opset < 8: might need to wrap concat in casts since only float is supported
        #if ctx.opset < 8:


    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)