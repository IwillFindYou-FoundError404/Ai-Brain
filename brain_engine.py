import numpy as np
import json
import os

class SpatialBrainEngine:
    def __init__(self, resolution=40):
        self.res = resolution # Độ phân giải ma trận ảnh (40x40 pixel để chạy mượt trên Render)
        
        # Định nghĩa 1 vật thể 3D trong không gian: Ví dụ một Hình Lập Phương (Cube) gồm 8 đỉnh
        self.cube_vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ]) * 1.5 # Phóng to kích thước hình lập phương

    def get_rotation_matrix(self, angle_x, angle_y, angle_z):
        """Tính toán ma trận xoay không gian 3 chiều"""
        rad_x, rad_y, rad_z = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
        
        # Xoay quanh trục X
        Rx = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
        # Xoay quanh trục Y
        Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
        # Xoay quanh trục Z
        Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
        
        return np.dot(Rz, np.dot(Ry, Rx))

    def render_spatial_frame(self, cam_z, rot_x, rot_y, rot_z):
        """Dựng một khung ảnh 2D từ không gian 3D dựa vào góc nhìn và khoảng cách"""
        # Tạo một khung ảnh trống (Màu đen = số 0)
        frame = np.zeros((self.res, self.res), dtype=np.uint8)
        
        # 1. Tính toán xoay vật thể trong không gian
        R = self.get_rotation_matrix(rot_x, rot_y, rot_z)
        rotated_vertices = np.dot(self.cube_vertices, R.T)
        
        # 2. Tịnh tiến vật thể ra xa theo trục Z dựa vào khoảng cách của Camera (cam_z)
        # Để vật thể nằm trước mặt camera
        translated_vertices = rotated_vertices.copy()
        translated_vertices[:, 2] += cam_z 
        
        # 3. Chiếu từ 3D xuống màn hình 2D (Perspective Projection)
        focal_length = self.res / 2 # Tiêu cự camera giả lập
        center = self.res // 2
        
        for vertex in translated_vertices:
            x_3d, y_3d, z_3d = vertex
            
            # Kiểm tra xem điểm này có nằm trong vùng nhìn thấy không (Z > 0 là trước mặt)
            if z_3d > 0.1:
                # Công thức thấu kính: Vật càng xa (z_3d lớn) thì tọa độ (x, y) trên màn hình càng nhỏ lại
                x_2d = int(center + (x_3d * focal_length) / z_3d)
                y_2d = int(center + (y_3d * focal_length) / z_3d)
                
                # Nếu điểm chiếu nằm trong phạm vi khung hình kích thước (res x res)
                if 0 <= x_2d < self.res and 0 <= y_2d < self.res:
                    # Vẽ điểm sáng lên ma trận ảnh (Giá trị độ sáng 255 = Trắng)
                    frame[y_2d, x_2d] = 255
                    
        return frame

    def generate_spatial_video(self, duration_seconds=7200, fps=24):
        """Sinh chuỗi video 24 FPS chuyển động không gian liên tục tối đa 2 tiếng không lo tràn RAM"""
        total_frames = duration_seconds * fps
        
        for frame_idx in range(total_frames):
            t = frame_idx / fps # Biến thời gian t
            
            # Giả lập vật thể tự xoay theo thời gian t trong không gian 3D
            rot_x = t * 30  # Xoay 30 độ mỗi giây
            rot_y = t * 45  # Xoay 45 độ mỗi giây
            rot_z = t * 15
            
            # Giả lập camera tiến lùi xa gần (Hiểu độ sâu không gian)
            # Khoảng cách biến thiên tuần hoàn từ 4.0 mét đến 7.0 mét
            cam_z = 5.5 + np.sin(t * 2) * 1.5 
            
            # Bộ não tiến hành render khung hình không gian
            frame_matrix = self.render_spatial_frame(cam_z, rot_x, rot_y, rot_z)
            
            yield {
                "frame": frame_idx,
                "timestamp": round(t, 3),
                "camera_distance_meters": round(float(cam_z), 2),
                "rotation_angles": [round(rot_x % 360, 1), round(rot_y % 360, 1)],
                "pixel_matrix": frame_matrix.tolist() # Xuất mảng pixel
            }

spatial_brain = SpatialBrainEngine(resolution=40)
