"""M1: connect to CoppeliaSim and read/write the arm joints."""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def connect():
    client = RemoteAPIClient()
    sim = client.require("sim")
    return client, sim
