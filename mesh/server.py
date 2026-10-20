# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

"""
Pure Standard Library HTTP REST Daemon for Federated Coordination.
Exposes endpoints for edge research node registration, update submission, and round aggregation.
"""

from __future__ import annotations
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from .coordinator import FederationCoordinator


class FederationHttpHandler(BaseHTTPRequestHandler):
    coordinator: Optional[FederationCoordinator] = None

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        coord = self.coordinator
        if not coord:
            self._set_headers(500)
            self.wfile.write(b'{"error": "Coordinator not initialized"}')
            return

        if self.path == "/health":
            self._set_headers(200)
            resp = {
                "status": "UP",
                "service": "federated-learning-privacy-mesh",
                "version": "1.0.0",
                "author": "Felipe Madison",
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path == "/model":
            self._set_headers(200)
            resp = {
                "current_round": coord.current_round,
                "dimension": coord.dimension,
                "model": coord.global_model.to_dict(),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path == "/metrics":
            self._set_headers(200)
            resp = {
                "current_round": coord.current_round,
                "registered_clients_count": len(coord.registered_clients),
                "cumulative_epsilon": coord.dp_mechanism.compute_cumulative_epsilon(),
                "delta": coord.dp_mechanism.target_delta,
                "strategy": coord.aggregation_strategy,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Not Found"}')

    def do_POST(self):
        coord = self.coordinator
        if not coord:
            self._set_headers(500)
            self.wfile.write(b'{"error": "Coordinator not initialized"}')
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._set_headers(400)
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        if self.path == "/register":
            client_id = data.get("client_id")
            institution = data.get("institution", "Unknown Institution")
            if not client_id:
                self._set_headers(400)
                self.wfile.write(b'{"error": "Missing client_id"}')
                return
            coord.register_client(client_id, institution)
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "REGISTERED", "client_id": client_id}).encode("utf-8"))

        elif self.path == "/submit_update":
            client_id = data.get("client_id")
            sample_count = int(data.get("sample_count", 1))
            params = data.get("parameters", [])
            try:
                coord.submit_client_update(client_id, sample_count, params)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ACCEPTED", "client_id": client_id}).encode("utf-8"))
            except ValueError as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path == "/finalize_round":
            summary = coord.finalize_round()
            self._set_headers(200)
            self.wfile.write(json.dumps(summary).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Not Found"}')


def run_daemon(port: int = 8088):
    coord = FederationCoordinator(dimension=4, aggregation_strategy="krum")
    FederationHttpHandler.coordinator = coord
    server = HTTPServer(("127.0.0.1", port), FederationHttpHandler)
    print(f"[Federated Learning Privacy Mesh] Serving on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    import sys
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run_daemon(p)
