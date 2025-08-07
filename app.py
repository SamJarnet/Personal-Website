from flask import Flask, render_template

app = Flask(__name__, static_url_path='/assets')


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/projects")
def projects():
    return render_template("projects.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    import sys
    host = "0.0.0.0"
    port = 5000

    # Optional: parse host and port from command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    host = args.host
    port = args.port

    app.run(host=host, port=port, debug=True)

