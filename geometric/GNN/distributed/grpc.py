import numpy as np
import grpc
from concurrent import futures
import gnn_rpc_pb2 as pb
import gnn_rpc_pb2_grpc as pb_rpc

class GNNDataHandler(pb_rpc.GNNDataHandlerServicer):
    def __init__(self, x, y, indptr, indices, nodes_from):
        super().__init__()
        # self.x = x
        # self.y = y
        # self.indptr = indptr
        # self.indices = indices
        # self.nodes_from = nodes_from
        num_nodes = x.shape[0]
        self.nodes = np.array(
            [
                pb.GNNNodeData(
                    feat=x[i],
                    label=y[i],
                    edge_nodes_id=indices[indptr[i]:indptr[i+1]],
                    edge_nodes_from=nodes_from[indptr[i]:indptr[i+1]]
                ) for i in range(num_nodes)
            ],
            dtype=object
        )

    def PullNodeFeature(self, request, context):
        index = request.index
        reply = pb.GNNDataReply(nodes=self.nodes[index])
        return reply

class GRPCGraphServer():
    def __init__(self, ip):
        MAX_MESSAGE_LENGTH = 256 * 1024 * 1024
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=_nrank),
            options=[('grpc.max_send_message_length', message_length),
                     ('grpc.max_receive_message_length', message_length)])
        self.server.add_insecure_port(ip)

    def start(self, handler):
        pb_rpc.add_GNNDataHandlerServicer_to_server(handler, self.server)
        self.server.start()

_server = None
_ip = None
_nrank = None
_rank = None
_stubs = None
_feat_len = None
message_length = 256 * 1024 * 1024

def grpc_init(hosts, ports, nrank, rank):
    global _ip, _server, _nrank, _rank
    _nrank = nrank
    _rank = rank
    _ip = ["{0}:{1}".format(hosts[i], ports[i]) for i in range(nrank)]
    _server = GRPCGraphServer(_ip[rank])

def grpc_stubs_init():
    global _stubs
    _stubs = []
    for i in range(_nrank):
        if i == _rank:
            _stubs.append(None)
        else:
            channel = grpc.insecure_channel(
                _ip[i],
                options=[('grpc.max_send_message_length', message_length),
                        ('grpc.max_receive_message_length', message_length)]
            )
            stub = pb_rpc.GNNDataHandlerStub(channel)
            _stubs.append(stub)

def grpc_upload(x, y, indptr, indices, nodes_from):
    global _feat_len
    _feat_len = x.shape[1]
    handler = GNNDataHandler(x, y, indptr, indices, nodes_from)
    _server.start(handler)

def grpc_download(nodes_id, nodes_from):
    num_nodes = len(nodes_id)
    # return empty arrays to soothe the caller
    if num_nodes == 0:
        return np.array([]).reshape(-1, _feat_len).astype(np.float32), \
               np.array([]).astype(np.int32),   \
               np.array([]).astype(np.long),   \
               np.vstack([[],[]]).astype(np.long)
    nodes = np.empty(shape=[num_nodes],dtype=object)
    # make requests and get all the data, reorder them to the original order
    import threading
    def remote_call(i):
        index = np.where(nodes_from==i)[0]
        if len(index) == 0:
            return
        request = pb.GNNDataRequest(index=nodes_id[index])
        reply = _stubs[i].PullNodeFeature(request)
        nodes[index] = reply.nodes
    threads = []
    for i in range(_nrank):
        th = threading.Thread(target=remote_call, args=[i])
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    # import time
    # start = time.time()
    feat = np.vstack([node.feat for node in nodes]).astype(np.float32)
    # print("feat", num_nodes, time.time() - start)
    label = np.array([node.label for node in nodes]).astype(np.int32)
    degree = np.array([len(node.edge_nodes_id) for node in nodes]).astype(np.long)
    # print("deg", num_nodes, time.time() - start)
    edges_id = np.concatenate([node.edge_nodes_id for node in nodes]).astype(np.long)
    edges_from = np.concatenate([node.edge_nodes_from for node in nodes]).astype(np.long)
    edges_data = np.vstack([edges_id, edges_from])
    # print(num_nodes, time.time() - start)
    return feat, label, degree, edges_data
