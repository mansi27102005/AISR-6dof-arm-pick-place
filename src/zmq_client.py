"""M1: connect to CoppeliaSim and read/write the arm joints."""


def connect(host="localhost", port=23004):
    """
    Connect to CoppeliaSim's ZeroMQ remote API.
    Requires CoppeliaSim running with the ZeroMQ Remote API add-on active
    and 'coppeliasim-zmqremoteapi-client' installed.
    """
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient(host=host, port=port)
        sim = client.require("sim")
        return client, sim
    except ImportError as e:
        raise ImportError(
            "coppeliasim-zmqremoteapi-client is not installed.\n"
            "To connect to CoppeliaSim, install it using: pip install coppeliasim-zmqremoteapi-client"
        ) from e


if __name__ == "__main__":
    try:
        client, sim = connect()
        print("Connected successfully to CoppeliaSim ZeroMQ API!")
    except Exception as exc:
        print(f"CoppeliaSim connection check: {exc}")
