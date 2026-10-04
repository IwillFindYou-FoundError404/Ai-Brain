from flask import Flask, request, jsonify, render_template, Response
import time
from agent.agent_loop import agent_loop
from providers.text_provider import text_provider
from providers.spatial_provider import spatial_provider

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/agent/run', methods=['POST'])
def agent_run():
    data = request.get_json() or {}
    msg = data.get('message', '')
    if not msg: return jsonify({'error': 'Thiếu tham số message'}), 400
    
    result = agent_loop.run_cycle(msg)
    return jsonify(result)

@app.route('/agent/learn', methods=['POST'])
def agent_learn():
    data = request.get_json() or {}
    text, label = data.get('text', ''), data.get('label', None)
    if label not in [0, 1] or not text: # Khắc phục hoàn toàn lỗi cú pháp mảng trống cũ
        return jsonify({'error': 'Cần nhập văn bản chữ và nhãn chính xác [0 hoặc 1]'}), 400
    text_provider.train_step(text, label)
    return jsonify({'success': True, 'status': 'Cập nhật ma trận nơ-ron Adam thành công!'})

@app.route('/agent/stream-video')
def stream_video():
    def video_generator():
        for f in range(172800): # 2 tiếng đồng hồ tương đương 172.800 khung hình ở tần số 24 FPS
            frame_data = spatial_provider.execute({"frame_idx": f})
            yield f"data: {{\\"frame\\": {f}, \\"camera_z\\": {frame_data['camera_z']}, \\"pixel_matrix\\": {frame_data['pixel_matrix']}}}\\n\\n"
            time.sleep(1.0 / 24.0) # Đồng bộ đúng 24 khung hình/giây
            if f >= 120: # Điểm chặn giao diện client tránh treo log trình duyệt
                yield "data: {\\"end_demo\\": true}\\n\\n"
                break
    return Response(video_generator(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
