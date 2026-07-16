import socket

import pytest

from verl.utils.net_utils import get_free_port


def test_get_free_port_reserves_port_exclusively() -> None:
    port, reserved_socket = get_free_port("127.0.0.1")
    contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        with pytest.raises(OSError):
            contender.bind(("127.0.0.1", port))
    finally:
        contender.close()
        reserved_socket.close()
