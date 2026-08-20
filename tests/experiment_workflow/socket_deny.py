# SPDX-License-Identifier: Apache-2.0

import socket


class DeniedSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network disabled by experiment workflow acceptance")


socket.socket = DeniedSocket
