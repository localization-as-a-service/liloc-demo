import numpy as np

from collections import deque

class TimestampArrayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque()

    def add_data(self, timestamp, array):
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.popleft()
        self.buffer.append((timestamp, array))

    def get_data(self):
        return list(self.buffer)

    def find_nearest(self, target_timestamp):
        nearest_pair = min(self.buffer, key=lambda pair: abs(pair[0] - target_timestamp))
        return nearest_pair
    
    def get_latest(self):
        return self.buffer[-1] if len(self.buffer) > 0 else np.array([0, 0])