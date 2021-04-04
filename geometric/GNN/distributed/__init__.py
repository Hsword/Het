from .ps import *

def is_grpc_avaiable():
    try:
        import gnn_rpc_pb2
        import gnn_rpc_pb2_grpc
    except ImportError:
        return False
    return True

if is_grpc_avaiable():
    from .grpc import *
