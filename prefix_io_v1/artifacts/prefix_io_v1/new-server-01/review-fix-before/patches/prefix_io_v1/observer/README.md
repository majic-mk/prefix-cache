# Observer preparation
No observer patch is applied to upstream execution yet. src/prefix_io_control/observation.py is an owner-thread-only bounded capture function, tested with CPU fixtures. It is intentionally not called by the native reactor. The mandatory wait bridge and scheduler generation adapter remain P2 work after P1 acceptance.
