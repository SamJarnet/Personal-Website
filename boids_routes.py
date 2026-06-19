from flask import Blueprint, jsonify, request, render_template
from boids_engine import Boids

boids_bp = Blueprint("boids", __name__)

# Persistent engine storage across network requests
_active_engine = None

@boids_bp.route("/boids")
def boids_page():
    return render_template("boids.html")

@boids_bp.route("/api/boids/simulate", methods=["POST"])
def simulate_boids():
    global _active_engine
    try:
        body = request.get_json() or {}
        cohesion = float(body.get("cohesion", 0.134))
        separation = float(body.get("separation", 0.045))
        alignment = float(body.get("alignment", 0.038))
        boid_count = int(body.get("boid_count", 40))
        reset_requested = bool(body.get("reset", False))
        
        # Instantiate a new engine ONLY if it doesn't exist, if boid count changed, or if a manual reset was triggered
        if (_active_engine is None or reset_requested or len(_active_engine.boids) != boid_count):
            _active_engine = Boids(
                cohesion=cohesion, 
                separation=separation, 
                alignment=alignment, 
                boid_count=boid_count, 
                map_size=25.0
            )
        else:
            # Dynamically update environmental forces on the flying swarm without resetting their coordinates!
            _active_engine.cohesion_strength = cohesion
            _active_engine.seperation_strength = separation
            _active_engine.alignment_strength = alignment
        
        # Calculate the next sequence starting exactly from where the last frame left off
        frames_data = _active_engine.run_simulation(total_frames=360)
        
        return jsonify({"status": "success", "frames": frames_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500