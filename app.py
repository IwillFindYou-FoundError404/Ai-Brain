from flask import Flask, request, jsonify, render_template, Response
import numpy as np
import time
from brain_engine import brain, spatial_brain, advanced_brain
from processor import processor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/think', methods=['POST'])
def think():
    data = request.get_json() or {}
    text = data.get('text', '')
    vector = processor.text_to_vector(text)
    prediction = brain.forward(vector)
    categories = ["Thân thiện / Vui vẻ", "Tiêu cực / Bực bội"]
    chosen_idx = int(np.argmax(prediction))
    return jsonify({'vector_representation': vector.tolist(), 'brain_decision': categories[chosen_idx]})

@app.route('/generate-art', methods=['POST'])
def generate_art():
    data = request.get_json() or {}
    prompt_text = data.get('text', 'vui')
    style_vector = processor.text_to_vector(prompt_text)[:5]
    latent_noise = np.random.randn(3)
    img_matrix = advanced_brain.generate_image_matrix(latent_noise, style_vector)
    return jsonify({'pixel_data': img_matrix.tolist()})

@app.route('/stream-spatial-video')
def stream_spatial_video():
    def generate_stream():
        video_stream = spatial_brain.generate_spatial_video(duration_seconds=7200, fps=24)
        for data in video_stream:
            yield f"data: {jsonify(data).get_data(as_text=True)}\n\n"
            time.sleep(1 / 24)
            if data["frame"] >= 100:  # Chặn demo giao diện chống lag trình duyệt client
                yield "data: {\"end_demo\": true}\n\n"
                break
    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
