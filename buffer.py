import numpy as np
from collections import deque
import time

class TimestampArrayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque()

    def add_data(self, timestamp, array):
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.popleft()  # Remove the oldest pair if the buffer is full
        self.buffer.append((timestamp, array))

    def get_data(self):
        return list(self.buffer)

    def find_nearest(self, target_timestamp):
        nearest_pair = min(self.buffer, key=lambda pair: abs(pair[0] - target_timestamp))
        return nearest_pair

# Example usage:
if __name__ == "__main__":
    buffer = TimestampArrayBuffer(max_buffer_size=10)

    for i in range(10):
        timestamp = time.time()  # Current timestamp
        data_array = np.random.rand(3, 3)  # Example numpy array

        buffer.add_data(timestamp, data_array)
        print(f"Added data at timestamp {timestamp}")
        time.sleep(0.2)

    stored_data = buffer.get_data()
    for timestamp, data in stored_data:
        print(f"Timestamp: {timestamp}, Data: {data}")

    target_timestamp = time.time() - 2  # Example target timestamp
    nearest_data = buffer.find_nearest(target_timestamp)
    print(f"Nearest object to timestamp {target_timestamp}: Timestamp={nearest_data[0]}, Data={nearest_data[1]}")
