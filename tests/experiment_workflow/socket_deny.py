import socket

class DeniedSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network disabled by experiment workflow acceptance")

socket.socket = DeniedSocket
