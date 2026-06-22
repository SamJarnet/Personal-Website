import copy
from flask import Blueprint, jsonify, request, render_template
from engines.boids_engine import Boids

boids_bp = Blueprint("boids", __name__)

_active_engine = None
_cached_frames = None
_cached_engine = None

def _precompute_boids():
    global _cached_frames, _cached_engine
    print("Pre-computing initial Boids frames...")
    engine = Boids(cohesion=0.134, separation=0.045, alignment=0.038, boid_count=50, map_size=25.0)
    _cached_frames = engine.run_simulation(total_frames=360, run_learning=False)
    _cached_engine = engine 
    print("Boids pre-computation complete!")

_precompute_boids()

@boids_bp.route("/boids")
def boids_page():
    return render_template("boids.html")

@boids_bp.route("/api/boids/simulate", methods=["POST"])
def simulate_boids():
    global _active_engine, _cached_frames, _cached_engine
    try:
        body = request.get_json() or {}
        cohesion = float(body.get("cohesion", 0.134))
        separation = float(body.get("separation", 0.045))
        alignment = float(body.get("alignment", 0.038))
        boid_count = int(body.get("boid_count", 50))
        reset_requested = bool(body.get("reset", False))
        run_learning = bool(body.get("learning", False))
        
        # Direct Cache Intercept: Only allowed if learning is turned off
        if reset_requested and boid_count == 50 and not run_learning and _cached_frames is not None:
            _active_engine = copy.deepcopy(_cached_engine)
            return jsonify({"status": "success", "frames": _cached_frames})

        # Base engine state instantiation rules
        if (_active_engine is None or reset_requested or len(_active_engine.boids) != boid_count):
            _active_engine = Boids(
                cohesion=cohesion, 
                separation=separation, 
                alignment=alignment, 
                boid_count=boid_count, 
                map_size=25.0
            )
        else:
            # If manual adjustments are sent while learning is disabled, honor user parameters
            if not run_learning:
                _active_engine.cohesion_strength = cohesion
                _active_engine.seperation_strength = separation
                _active_engine.alignment_strength = alignment
        
        # Compute the frames tracking the adaptive learning flag
        frames_data = _active_engine.run_simulation(total_frames=360, run_learning=run_learning)
        
        return jsonify({"status": "success", "frames": frames_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500