from flask import Blueprint, jsonify, request, render_template
from engines.boids_engine import Boids

boids_bp = Blueprint("boids", __name__)

# Active engine instance maintained across sequential steps
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
        boid_count = int(body.get("boid_count", 50))
        reset = bool(body.get("reset", False))
        learning = bool(body.get("learning", False))

        # Initialize new engine if reset requested, uninitialized, or count changed
        current_count = len(_active_engine.boids) if _active_engine else 0
        if _active_engine is None or reset or current_count != boid_count:
            _active_engine = Boids(
                cohesion=cohesion, 
                separation=separation, 
                alignment=alignment, 
                boid_count=boid_count, 
                map_size=25.0
            )
        else:
            if not learning:
                _active_engine.cohesion_strength = cohesion
                _active_engine.seperation_strength = separation
                _active_engine.alignment_strength = alignment

        frames_data = _active_engine.run_simulation(total_frames=360, run_learning=learning)
        return jsonify({"status": "success", "frames": frames_data})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500