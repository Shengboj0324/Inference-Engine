"""Public-API contract freeze tests.

These tests pin the FastAPI route surface (paths, HTTP methods, websocket
endpoints) and the wire-shape of request/response Pydantic models that the
upcoming desktop UI sidecar will consume.

A failure here is intentional: it means a downstream UI build will break if
the change is shipped unchanged.  When a breaking change is required:

1. Update the snapshot in the corresponding test module.
2. Bump the contract version stamp.
3. Update the desktop UI client to match.
"""
