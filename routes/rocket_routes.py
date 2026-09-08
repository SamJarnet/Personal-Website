import os
import torch
import numpy as np
from flask import Blueprint, jsonify, request, render_template
from stable_baselines3 import PPO

# 2D Engine & Model Imports
from engines.physics_engine import RocketPhysics as RocketPhysics2D
from agent import RocketNetwork

# 3D Engine Import (Saved as physics_3d.py)
from engines.physics_3d import RocketPhysics as RocketPhysics3D

rocket_bp = Blueprint("rocket", __name__)

device = torch.device("cpu")

# --- 2D MODEL SETUP (DQN) ---
STATE_DIM_2D, ACTION_DIM_2D = 6, 6
MODEL_PATH_2D = "rocket_dqn.pth"
model_2d = RocketNetwork(STATE_DIM_2D, ACTION_DIM_2D).to(device)
if os.path.exists(MODEL_PATH_2D):
    model_2d.load_state_dict(torch.load(MODEL_PATH_2D, map_location=device))
    model_2d.eval()

# --- 3D MODEL SETUP (PPO) ---
MODEL_PATH_3D = "rocket_ppo_model"
model_3d = None
if os.path.exists(MODEL_PATH_3D) or os.path.exists(f"{MODEL_PATH_3D}.zip"):
    try:
        model_3d = PPO.load(MODEL_PATH_3D, device="cpu")
        print("Loaded 3D PPO model successfully.")
    except Exception as e:
        print(f"Error loading 3D PPO model: {e}")

# Persistent simulation instances
active_physics_2d = RocketPhysics2D()
active_physics_3d = RocketPhysics3D()

def get_obs_2d(physics):
    return np.array([
        physics.pos[0] / 10.0,
        physics.pos[1] / 20.0,
        physics.vel[0] / 5.0,
        physics.vel[1] / 5.0,
        physics.angle / np.pi,
        physics.angular_vel / 2.0
    ], dtype=np.float32)

def get_obs_3d(physics):
    target = physics.landing_pad
    pos = physics.pos
    vel = physics.vel

    return np.array([
        (pos[0] - target[0]) / 10.0,
        (pos[1] - target[1]) / 20.0,
        (pos[2] - target[2]) / 10.0,
        vel[0] / 5.0,
        vel[1] / 5.0,
        vel[2] / 5.0,
        physics.angle[0] / np.pi,
        physics.angle[1] / np.pi,
        physics.angle[2] / np.pi,
        physics.angular_vel[0] / 2.0,
        physics.angular_vel[1] / 2.0,
        physics.angular_vel[2] / 2.0,
    ], dtype=np.float32)

@rocket_bp.route("/rocket")
def rocket_page():
    return render_template("rocket.html")

@rocket_bp.route("/api/rocket/simulate", methods=["POST"])
def simulate_rocket():
    global active_physics_2d, active_physics_3d
    try:
        body = request.get_json() or {}
        dim = body.get("dim", "3d")  # "3d" or "2d"
        mode = body.get("mode", "manual")
        is_reset = body.get("reset", False)

        # ==================== 3D SIMULATION ====================
        if dim == "3d":
            if is_reset:
                start_y = np.random.uniform(10.0, 14.0)
                start_x = np.random.uniform(4.0, 8.0)
                start_z = np.random.uniform(-2.0, 2.0)
                active_physics_3d.reset(pos=(start_x, start_y, start_z))

            action_name = "HUMAN MANUAL PILOT"

            if mode == "ai" and model_3d is not None:
                obs = get_obs_3d(active_physics_3d)
                action, _ = model_3d.predict(obs, deterministic=False)
                crashed, landed = active_physics_3d.step(action)
                action_name = "STABLE-BASELINES3 PPO AI"
            else:
                manual_action = body.get("action", [0.0, 0.0, 0.0])
                crashed, landed = active_physics_3d.step(np.array(manual_action, dtype=np.float32))

            frame_data = {
                "dim": "3d",
                "pos": [float(p) for p in active_physics_3d.pos],
                "vel": [float(v) for v in active_physics_3d.vel],
                "angle": [float(a) for a in active_physics_3d.angle],
                "thrusting": bool(active_physics_3d.thrusting),
                "crashed": bool(crashed),
                "landed": bool(landed),
                "action_name": action_name,
                "landing_pad": [float(p) for p in active_physics_3d.landing_pad]
            }

        # ==================== 2D SIMULATION ====================
        else:
            if is_reset:
                start_y = np.random.uniform(12.0, 15.0)
                start_x = np.random.uniform(3.0, 7.0)
                active_physics_2d.reset(pos=(start_x, start_y))

            action_name = "IDLE"

            if mode == "ai":
                obs = get_obs_2d(active_physics_2d)
                state_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model_2d(state_t)
                    action = torch.argmax(q_values).item()
                crashed, landed = active_physics_2d.step(action)
                action_name = "PYTORCH DQN AI"
            elif mode == "traditional":
                crashed, landed = active_physics_2d.step_traditional()
                action_name = "TRADITIONAL AUTOPILOT"
            else:
                manual_action = int(body.get("action", 0))
                crashed, landed = active_physics_2d.step(manual_action)
                action_name = "HUMAN MANUAL PILOT"

            frame_data = {
                "dim": "2d",
                "x": float(active_physics_2d.pos[0]),
                "y": float(active_physics_2d.pos[1]),
                "angle": float(active_physics_2d.angle),
                "vx": float(active_physics_2d.vel[0]),
                "vy": float(active_physics_2d.vel[1]),
                "thrusting": bool(active_physics_2d.thrusting),
                "crashed": bool(crashed),
                "landed": bool(landed),
                "action_name": action_name,
                "landing_pad": [float(active_physics_2d.landing_pad[0]), float(active_physics_2d.landing_pad[1])]
            }

        return jsonify({"status": "success", "frame": frame_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500