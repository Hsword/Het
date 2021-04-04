from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["ReduceMeanOp"],onnx_op=["ReduceMean"])
@athena_op(["ReduceSumOp"],onnx_op=["ReduceSum"])
class ReduceMean(general.PassOp):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        keepdims=node.get_attr_value('keepdims',None)
        assert keepdims is not None
        node.set_attr("keepdims",keepdims[0])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


