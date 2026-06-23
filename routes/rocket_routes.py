import copy
from flask import Blueprint, jsonify, request, render_template
from engines.rocket_engine import RocketLandingEngine

rocket_bp = Blueprint("rocket", __name__)

_active_rocket = None
_cached_frames = None
_cached_rocket = None

def _precompute_rocket():
    global _cached_frames, _cached_rocket
    print("Pre-computing initial rocket landing frames...")
    engine = RocketLandingEngine(initial_pos=[2.0, 1.0], initial_vel=[0.0, 0.0])
    # Compute initial baseline down-drop trail cleanly
    _cached_frames = engine.run_simulation(total_frames=120, thrusting=False, landing=False)
    _cached_rocket = engine
    print("Rocket pre-computation complete!")

_precompute_rocket()

@rocket_bp.route("/rocket")
def rocket_page():
    return render_template("rocket.html")

@rocket_bp.route("/api/rocket/simulate", methods=["POST"])
def simulate_rocket():
    global _active_rocket, _cached_frames, _cached_rocket
    try:
        body = request.get_json() or {}
        reset_requested = bool(body.get("reset", False))
        thrusting = bool(body.get("thrusting", False))
        landing = bool(body.get("landing", False))
        
        if reset_requested and not thrusting and not landing and _cached_frames is not None:
            _active_rocket = copy.deepcopy(_cached_rocket)
            return jsonify({"status": "success", "frames": _cached_frames})

        if _active_rocket is None or reset_requested:
            _active_rocket = RocketLandingEngine(initial_pos=[2.0, 1.0], initial_vel=[0.0, 0.0])
        
        # FIX: Drop frames generated down to 1 so the frontend UI can stream real-time choices
        frames_data = _active_rocket.run_simulation(
            total_frames=1, 
            thrusting=thrusting, 
            landing=landing
        )
        
        return jsonify({"status": "success", "frames": frames_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500