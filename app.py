from flask import Flask, request, jsonify, render_template, Response
import time
from brain_engine import spatial_brain # Import bộ não không gian mới

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream-spatial-video')
def stream_spatial_video():
    """Luồng phát Video 3D mô phỏng không gian thực tế với tốc độ 24 FPS"""
    def generate_stream():
        # Gọi Generator sinh cuốn chiếu, chạy liên tục tối đa 2 tiếng
        video_stream = spatial_brain.generate_spatial_video(duration_seconds=7200, fps=24)
        
        for data in video_stream:
            # Gửi gói dữ liệu khung hình dạng JSON qua stream
            yield f"data: {jsonify(data).get_data(as_text=True)}\n\n"
            
            # Giới hạn chuẩn tốc độ phát 24 khung hình trên giây
            time.sleep(1 / 24)
            
            # Chặn demo ở frame 100 trên giao diện để tránh trình duyệt của bạn bị quá tải log
            if data["frame"] >= 100:
                yield "data: {\"end_demo\": true}\n\n"
                break

    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
