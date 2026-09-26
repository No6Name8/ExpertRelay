"""The Manager: understands every device in the cluster, decides where each
expert lives, and dispatches each request to the right one.

Today it holds only the machine profile (profile.py). The single-device
runtime (expertrelay.runtime) reads every expert from local disk; placement
across devices is not implemented yet. See docs/limitations.md.
"""
