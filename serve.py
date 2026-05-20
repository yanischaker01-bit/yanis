#!/usr/bin/env python3
"""
serve.py – Serveur local pour la carte LGV SEA
Ouvre automatiquement http://localhost:8050 dans le navigateur.
"""
import http.server
import webbrowser
import threading
import os
from pathlib import Path

PORT = 8050
os.chdir(Path(__file__).parent)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silencieux

print(f"  Serveur démarré : http://localhost:{PORT}")
print("  Ctrl+C pour arrêter")
threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
with http.server.HTTPServer(("", PORT), Handler) as srv:
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Serveur arrêté.")
