"""Process entrypoint — launches the Flask development server."""

from presentation.app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
