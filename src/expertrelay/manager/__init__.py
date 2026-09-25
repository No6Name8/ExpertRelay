"""The Manager: understands every device in the cluster, decides where each
expert lives, and dispatches each request to the right one.

Today's implementation (coordinator.py) is a minimal, concrete instance of
that idea for exactly two devices with a static expert assignment -- not
yet the general hot/cold placement + prediction system described in
README.md. See docs/limitations.md.
"""
