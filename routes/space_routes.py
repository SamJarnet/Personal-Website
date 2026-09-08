import copy
from flask import Blueprint, jsonify, request, render_template
from engines.space_engine import SpaceEngine

space_bp = Blueprint("space", __name__)

_active_engine = None
_cached_frames = None
_cached_engine = None

def _precompute_space():
    """Pre-computes initial frames on startup."""
    global _cached_frames, _cached_engine
    try:
        engine = SpaceEngine()
        _cached_frames = engine.run_simulation(total_frames=120, dt=0.7)
        _cached_engine = engine 
    except Exception as e:
        print(f"space pre-computation failed: {e}")

_precompute_space()

@space_bp.route("/space")
def space_page():
    return render_template("space.html")

@space_bp.route("/api/space/simulate", methods=["POST"])
def simulate_space():
    global _active_engine, _cached_frames, _cached_engine
    
    try:
        body_data = request.get_json() or {}
        dt = float(body_data.get("dt", 0.7))
        action = body_data.get("action")
        reset_requested = bool(body_data.get("reset", False))
        
        if reset_requested and _cached_frames is not None:
            _active_engine = copy.deepcopy(_cached_engine)
            return jsonify({"status": "success", "frames": _cached_frames})

        if _active_engine is None or reset_requested:
            _active_engine = SpaceEngine()


        frames_data = _active_engine.run_simulation(total_frames=60, dt=dt)
        
        return jsonify({
            "status": "success", 
            "frames": frames_data
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500