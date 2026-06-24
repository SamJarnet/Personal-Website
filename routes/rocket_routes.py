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
    # Spawn at high altitude by default
    engine = RocketLandingEngine(initial_pos=[2.0, 25.0], initial_vel=[0.0, 0.0])
    _cached_frames = engine.run_simulation(total_frames=1, thrusting=False, landing=False)
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
        turn_left = bool(body.get("turn_left", False))
        turn_right = bool(body.get("turn_right", False))
        landing = bool(body.get("landing", False))
        
        if reset_requested and _cached_rocket is not None:
            _active_rocket = copy.deepcopy(_cached_rocket)
            # Regenerate dynamic fresh frames from starting state
            frames_data = _active_rocket.run_simulation(total_frames=1, thrusting=False, landing=False)
            return jsonify({"status": "success", "frames": frames_data})

        if _active_rocket is None or reset_requested:
            _active_rocket = RocketLandingEngine(initial_pos=[2.0, 25.0], initial_vel=[0.0, 0.0])
        
        frames_data = _active_rocket.run_simulation(
            total_frames=1, 
            thrusting=thrusting, 
            turn_left=turn_left,
            turn_right=turn_right,
            landing=landing
        )
        
        return jsonify({"status": "success", "frames": frames_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500