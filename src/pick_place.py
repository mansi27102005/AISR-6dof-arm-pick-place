"""M4: pick-and-place state machine (approach, grasp, lift, move, place, retreat)."""
from zmq_client import connect

if __name__ == "__main__":
    client, sim = connect()
    print("Connected to CoppeliaSim")
