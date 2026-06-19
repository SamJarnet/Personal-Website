from flask import Blueprint, render_template, request, Response, jsonify
import cv2
import numpy as np
from engines.motion_engine import MotionDetectorEngine

motion_bp = Blueprint('motion', __name__)

# Global engine instance
engine = MotionDetectorEngine()

# --- MAIN PAGE ROUTE ---
@motion_bp.route('/motion')  # Changed from '/' to '/motion'
def index():
    """Main motion detection page."""
    return render_template('motion.html')

# --- API ROUTES ---
@motion_bp.route('/process_frame', methods=['POST'])
def process_frame():
    """Receive a frame from the client, process it, return the processed image."""
    file = request.files.get('frame')
    if not file:
        return "No frame", 400

    # Decode JPEG/PNG to BGR
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid image", 400

    mode = request.args.get('mode', 'overlay')
    processed = engine.process_frame(frame, mode)

    # Encode as JPEG and return
    _, jpeg = cv2.imencode('.jpg', processed)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@motion_bp.route('/status', methods=['GET'])
def motion_status():
    """Return whether motion is currently detected."""
    return jsonify({'movement': engine.get_motion_status()})

@motion_bp.route('/threshold', methods=['POST'])
def set_threshold():
    """Update the motion detection threshold."""
    data = request.get_json()
    if 'value' not in data:
        return jsonify({'error': 'Missing value'}), 400
    engine.set_threshold(data['value'])
    return jsonify({'current_threshold': engine.threshold})