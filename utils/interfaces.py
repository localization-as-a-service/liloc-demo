import requests
import numpy as np


class GlobalLiDARInterface:
    def __init__(self, base_url):
        self.base_url = base_url
        
    def start(self):
        response = requests.get(f"{self.base_url}/start")
        return response.text == "OK"
    
    def stop(self):
        response = requests.get(f"{self.base_url}/stop")
        return response.text == "OK"
    
    def get_global_pcd(self, timestamp):
        response = requests.get(f"{self.base_url}/global-pcd?timestamp={timestamp}")
        data = response.json()
        return np.array(data)
    

class GlobalRegistrationInterface:
    def __init__(self, base_url):
        self.base_url = base_url

    def global_registration(self, source, target):
        response = requests.get(f"{self.base_url}/register", json={
            "source": source.astype(np.float16).tolist(),
            "target": target.astype(np.float16).tolist()
        })

        return np.array(response.json())
    

class LiLOCServiceInterface:
    def __init__(self, base_url, session_id):
        self.base_url = base_url
        self.session_id = session_id
        
    def start(self):
        response = requests.get(f"{self.base_url}/session/start?session_id={self.session_id}")
        return response.text == "OK"
    
    def stop(self):
        response = requests.get(f"{self.base_url}/session/stop?session_id={self.session_id}")
        return response.text == "OK"
    
    def get_global_pcd(self, timestamp):
        response = requests.get(f"{self.base_url}/global-pcd?timestamp={timestamp}")
        data = response.json()
        return np.array(data)
    
    def update_local(self, timestamp, transformation_matrix):
        response = requests.get(f"{self.base_url}/session/local?session_id={self.session_id}", json={
            "timestamp": timestamp,
            "transformation_matrix": transformation_matrix.astype(np.float16).tolist()
        })
        
        return np.array(response.json())
    
    def update_global(self, timestamp, source):
        response = requests.get(f"{self.base_url}/session/global?session_id={self.session_id}", json={
            "timestamp": timestamp,
            "source": source.astype(np.float16).tolist()
        })
        
        return np.array(response.json())
    

    def get_current_view(self):
        response = requests.get(f"{self.base_url}/session/current?session_id={self.session_id}")
        return response.json()
