from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["WhereOp"],onnx_op=["Where"])
class Where():
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        assert False,"This version of the operator has been available since version 9 of the default ONNX operator set"
        pass

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass