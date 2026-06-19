from flask import Blueprint, render_template, Response, request, jsonify
from engines.motion_engine import MotionDetectorEngine

motion_bp = Blueprint("motion", __name__)

detector = MotionDetectorEngine(camera_index=0)

@motion_bp.route("/motion")
def motion_page():
    return render_template("motion.html")

@motion_bp.route("/api/motion/feed")
def video_feed():
    mode = request.args.get("mode", "overlay")
    return Response(
        detector.show_camera(mode=mode),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@motion_bp.route("/api/motion/stop", methods=["POST"])
def stop_camera():
    released = detector.release_camera()
    return jsonify({
        "status": "success" if released else "idle",
        "message": "Camera hardware released."
    })

@motion_bp.route("/api/motion/status", methods=["GET"])
def motion_status():
    return jsonify({
        "movement": getattr(detector, "is_moving", False)
    })

# NEW: Receiver API endpoint to ingest live frontend slider adjustments
@motion_bp.route("/api/motion/threshold", methods=["POST"])
def update_threshold():
    data = request.get_json() or {}
    new_val = data.get("value")
    
    if new_val is not None:
        try:
            # Force constraints (OpenCV pixel thresholds must live between 1 and 255)
            detector.threshold = max(1, min(255, int(new_val)))
            return jsonify({"status": "updated", "current_threshold": detector.threshold})
        except ValueError:
            return jsonify({"error": "Invalid numerical threshold supplied"}), 400
            
    return jsonify({"error": "Missing value parameter"}), 400