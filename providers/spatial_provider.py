import numpy as np
from providers.base_provider import BaseProvider

class SpatialProvider(BaseProvider):
    def __init__(self, resolution=40):
        self.res = resolution
        # Tọa độ vật thể hình khối lập phương 3D trong không gian thực
        self.vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * 1.5

    def execute(self, data):
        frame_idx = data.get('frame_idx', 0)
        t = frame_idx / 24
        frame = np.zeros((self.res, self.res), dtype=np.uint8)
        
        # Thiết lập ma trận xoay góc Pitch và Yaw toán học liên tục
        rx, ry = np.radians(t * 30), np.radians(t * 45)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        rotated = np.dot(self.vertices, np.dot(Ry, Rx).T)
        
        # Hiểu độ sâu không gian: Camera tiến lùi tuần hoàn từ trục Z dài 4m đến 7m
        cam_z = 5.5 + np.sin(t * 2) * 1.5
        translated = rotated.copy()
        translated[:, 2] += cam_z
        
        # Phép chiếu phối cảnh thấu kính (Perspective Projection)
        focal, center = self.res / 2, self.res // 2
        for vertex in translated:
            x_3d, y_3d, z_3d = vertex
            if z_3d > 0.1:
                # Vật càng ở xa camera (Z lớn) thì ảnh chiếu lên màn hình càng co nhỏ lại
                x_2d = int(center + (x_3d * focal) / z_3d)
                y_2d = int(center + (y_3d * focal) / z_3d)
                if 0 <= x_2d < self.res and 0 <= y_2d < self.res:
                    frame[y_2d, x_2d] = 255
                    
        return {"frame": frame_idx, "camera_z": float(cam_z), "pixel_matrix": frame.tolist()}

spatial_provider = SpatialProvider()
