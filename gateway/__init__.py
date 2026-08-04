"""Compatibility namespace for former gateway-owned core helpers.

Messaging adapters, delivery, and service lifecycle code are deliberately not
part of the training runtime.  Keep this package path stable for the session
context and small process/media helpers still imported by the agent core.
"""
