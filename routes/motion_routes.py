from flask import Blueprint, render_template, request, Response, jsonify
import cv2
import numpy as np
from engines.motion_engine import MotionDetectorEngine

motion_bp = Blueprint('motion', __name__)
engine = MotionDetectorEngine()

@motion_bp.route('/motion')
def index():
    """Main motion detection page."""
    return render_template('motion.html')

@motion_bp.route('/process_frame', methods=['POST'])
def process_frame():
    """Receive frame, process motion detection, and attach motion status in header."""
    file = request.files.get('frame')
    if not file:
        return "No frame", 400

    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid image", 400

    mode = request.args.get('mode', 'overlay')
    processed = engine.process_frame(frame, mode)

    _, jpeg = cv2.imencode('.jpg', processed)
    
    # Return processed frame with motion status included in the HTTP headers
    response = Response(jpeg.tobytes(), mimetype='image/jpeg')
    response.headers['X-Motion-Detected'] = str(engine.get_motion_status())
    return response

@motion_bp.route('/threshold', methods=['POST'])
def set_threshold():
    """Update motion detection sensitivity threshold."""
    data = request.get_json() or {}
    if 'value' in data:
        engine.set_threshold(data['value'])
    return jsonify({'current_threshold': engine.threshold})