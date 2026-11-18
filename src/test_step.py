import numpy as np
import zmq_client as z

client, sim = z.connect()
z.start_simulation(client, sim)

q = np.array([0.3, -1.2, 1.0, -1.5, -1.5, 0.2])
z.set_joint_angles(sim, q)
print("right after set :", np.round(z.get_joint_angles(sim), 3))
for _ in range(10):
    client.step()
print("after 10 steps  :", np.round(z.get_joint_angles(sim), 3))
print("target          :", np.round(q, 3))

z.stop_simulation(client, sim)
client.setStepping(False)