import time
import open3d
import zmq
import numpy as np
import cv2


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md["dtype"])
    return A.reshape(md["shape"])


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    while True:
        try:
            depth_frame = recv_array(socket)
            cv2.imshow("Depth Camera", depth_frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        except KeyboardInterrupt:
            break
        
    socket.close()

if __name__ == '__main__':
    main()

