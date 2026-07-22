import copy
from flask import Blueprint, jsonify, request, render_template
from engines.boids_engine import Boids

boids_bp = Blueprint("boids", __name__)

_active_engine = None
_cached_frames = None
_cached_engine = None

def _precompute_boids():
    """
    Pre-computes initial default simulation frames on startup 
    to accelerate initial web page render times.
    """
    global _cached_frames, _cached_engine
    try:
        print("Pre-computing initial Boids frames...")
        engine = Boids(
            cohesion=0.134, 
            separation=0.045, 
            alignment=0.038, 
            boid_count=50, 
            map_size=25.0
        )
        _cached_frames = engine.run_simulation(total_frames=360, run_learning=False)
        _cached_engine = engine 
        print("Boids pre-computation complete!")
    except Exception as e:
        print(f"Warning: Boids pre-computation skipped or failed: {e}")

# Pre-compute initial frame buffer when blueprint initializes
_precompute_boids()


@boids_bp.route("/boids")
def boids_page():
    """Renders the main Boids simulation template."""
    return render_template("boids.html")


@boids_bp.route("/api/boids/simulate", methods=["POST"])
def simulate_boids():
    """
    API Endpoint called by boids.html when using the Python Backend engine.
    Handles dynamic parameter tuning, cache hits, and adaptive RL learning flags.
    """
    global _active_engine, _cached_frames, _cached_engine
    
    try:
        body = request.get_json() or {}
        
        # Parse payload inputs with fallback default parameters
        cohesion = float(body.get("cohesion", 0.134))
        separation = float(body.get("separation", 0.045))
        alignment = float(body.get("alignment", 0.038))
        boid_count = int(body.get("boid_count", 50))
        reset_requested = bool(body.get("reset", False))
        run_learning = bool(body.get("learning", False))
        
        # 1. Direct Cache Intercept:
        # Re-use pre-computed startup frames if resetting with defaults and learning is off
        if reset_requested and boid_count == 50 and not run_learning and _cached_frames is not None:
            _active_engine = copy.deepcopy(_cached_engine)
            return jsonify({
                "status": "success", 
                "frames": _cached_frames
            })

        # 2. Engine State Lifecycle Rules:
        # Instantiate engine if none exists, reset requested, or boid count changed
        current_count = len(_active_engine.boids) if (_active_engine and hasattr(_active_engine, 'boids')) else 0
        
        if (_active_engine is None or reset_requested or current_count != boid_count):
            _active_engine = Boids(
                cohesion=cohesion, 
                separation=separation, 
                alignment=alignment, 
                boid_count=boid_count, 
                map_size=25.0
            )
        else:
            # Dynamically update flocking weights if manually adjusted WITHOUT RL learning active
            if not run_learning:
                if hasattr(_active_engine, 'cohesion_strength'):
                    _active_engine.cohesion_strength = cohesion
                if hasattr(_active_engine, 'seperation_strength'):
                    _active_engine.seperation_strength = separation
                if hasattr(_active_engine, 'alignment_strength'):
                    _active_engine.alignment_strength = alignment
        
        # 3. Compute Frame Sequence:
        frames_data = _active_engine.run_simulation(total_frames=360, run_learning=run_learning)
        
        return jsonify({
            "status": "success", 
            "frames": frames_data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500