"""
rl_routes.py
Flask blueprint handling the Reinforcement Learning Lab page and its API
endpoints. Wires up the user-defined bandit/gridworld engines, same pattern
as trading_routes.py uses for the strategy engine.
"""

from flask import Blueprint, jsonify, request, render_template

import engines.rl.bandit_engine as bandit_engine
import engines.rl.bandit_ucb_engine as bandit_ucb_engine
import engines.rl.bandit_bayesian_engine as bandit_bayesian_engine
import engines.rl.bandit_contextual_engine as bandit_contextual_engine
import engines.rl.gridworld_engine as gridworld_engine
import engines.rl.gridworld_bellman_engine as gridworld_bellman_engine
import engines.rl.gridworld_continuous_engine as gridworld_continuous_engine
import engines.rl.gridworld_continuous_walls_engine as gridworld_continuous_walls_engine

rl_bp = Blueprint("rl", __name__)


# ── page ────────────────────────────────────────────────────────────────────

@rl_bp.route("/rl")
def rl_page():
    return render_template("rl.html")


# ── multi-armed bandits ──────────────────────────────────────────────────────

BANDIT_ENGINES = {
    "epsilon_greedy": bandit_engine,
    "ucb":            bandit_ucb_engine,
    "bayesian":       bandit_bayesian_engine,
}

@rl_bp.route("/api/rl/bandit/run", methods=["POST"])
def run_bandit():
    try:
        body = request.get_json() or {}
        variant = body.get("variant", "epsilon_greedy")

        engine = BANDIT_ENGINES.get(variant)
        if engine is None:
            return jsonify({"error": f"Unknown bandit variant '{variant}'"}), 400

        # Clamp request-controlled values — these loops run synchronously
        # inside the request, so an unbounded eps/trials value from the
        # client could tie up a worker for a long time.
        eps = int(body.get("episodes", 10000))
        eps = max(100, min(eps, 200000))
        n = int(body.get("trials", 50))
        n = max(1, min(n, 200))

        result = engine.run_simulation(eps, n)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rl_bp.route("/api/rl/bandit/contextual/run", methods=["POST"])
def run_contextual_bandit():
    try:
        body = request.get_json() or {}

        # learn()'s inner loop also runs `eps` times, so cost is O(eps^2) —
        # clamp tighter than the other bandit variants to avoid a long-running
        # request.
        eps = int(body.get("episodes", 1000))
        eps = max(10, min(eps, 3000))

        result = bandit_contextual_engine.run_simulation(eps)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── gridworlds ────────────────────────────────────────────────────────────────

GRIDWORLD_ENGINES = {
    "qlearning":        gridworld_engine,
    "bellman":          gridworld_bellman_engine,
    "continuous":       gridworld_continuous_engine,
    "continuous_walls": gridworld_continuous_walls_engine,
}

@rl_bp.route("/api/rl/gridworld/run", methods=["POST"])
def run_gridworld():
    try:
        body = request.get_json() or {}
        variant = body.get("variant", "qlearning")

        engine = GRIDWORLD_ENGINES.get(variant)
        if engine is None:
            return jsonify({"error": f"Unknown gridworld variant '{variant}'"}), 400

        eps = int(body.get("episodes", 1000))
        eps = max(10, min(eps, 20000))

        result = engine.run_simulation(eps)
        result["variant"] = variant
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
