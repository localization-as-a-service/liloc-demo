import zmq
import time
import numpy as np
import multiprocessing as mp
# import signal
# import sys

# def signal_handler(sig, frame):
#     print("Keyboard interrupt received. Exiting...")
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)

class GPCRequester(mp.Process):
    def __init__(self, queue: mp.Queue, address: str = "localhost", port: int = 5555):
        super(GPCRequester, self).__init__()
        self.queue = queue
        self.url = f"tcp://{address}:{port}"
        
    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect(self.url)
        print("Connected to GPC server")
        
        while True:
            try:
                data, timestamp = self.queue.get()
                socket.send_string("Send GPC")
            except KeyboardInterrupt:
                break
            
        socket.close()
        
        
class GPCServer(mp.Process):
    def __init__(self, address: str = "*", port: int = 5555):
        super(GPCServer, self).__init__()
        self.url = f"tcp://{address}:{port}"
        
    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.bind(self.url)
        print("GPC server started")
        
        while True:
            try:
                msg = socket.recv_string()
                print(msg)
            except KeyboardInterrupt:
                break
            
        socket.close()
        
def main():
    queue = mp.Queue()
    gpc_server = GPCServer()
    gpc_server.start()
    
    gpc_requester = GPCRequester(queue)
    gpc_requester.start()
    
    for i in range(10):
        queue.put((i, time.time()))
        time.sleep(1)
    
    gpc_requester.terminate()
    gpc_server.terminate()
    
    gpc_requester.join()
    gpc_server.join()        
    
    
if __name__ == "__main__":
    main()
