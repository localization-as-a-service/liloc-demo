import zmq
import time
import numpy as np
import utils.grid_search_rs_unopt as grid_search
import utils.registration as registration


# def receive_array(socket, flags=0, copy=True, track=False):
#     md = socket.recv_json(flags=flags)
    
#     source_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     source = np.frombuffer(memoryview(source_buffer), dtype=md['dtype']).reshape(md['source'])
    
#     source_feat_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     source_feat = np.frombuffer(memoryview(source_feat_buffer), dtype=md['dtype']).reshape(md['source_feat'])
    
#     target_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     target = np.frombuffer(memoryview(target_buffer), dtype=md['dtype']).reshape(md['target'])
    
#     target_feat_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     target_feat = np.frombuffer(memoryview(target_feat_buffer), dtype=md['dtype']).reshape(md['target_feat'])
    
#     return source, source_feat, target, target_feat

def receive_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    return np.array(md['source']), np.array(md['source_feat']), np.array(md['target']), np.array(md['target_feat'])


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:5558")
    
    while True:
        try:
            source, source_feat, target, target_feat = receive_array(socket)
            print(f"Source: {source.shape} | Source Feat: {source_feat.shape} | Target: {target.shape} | Target Feat: {target_feat.shape}")
            source, target, result = grid_search.global_registration(source, source_feat, target, target_feat, cell_size=2, n_random=0.8, refine_enabled=True)
            print(result)
            registration.view(source, target, result.transformation)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()