from website import create_app
import argparse

app = create_app()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Grad Project")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)