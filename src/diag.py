import zmq_client as z

client, sim = z.connect()
z.start_simulation(client, sim)
h = z._handles(sim)

print("kinematic constant:", sim.jointmode_kinematic)
for j in h["joints"]:
    print(sim.getObjectAlias(j, 1),
          "| mode:", sim.getJointMode(j),
          "| ctrl mode:", sim.getObjectInt32Param(j, sim.jointintparam_dynctrlmode),
          "| motor on:", sim.getObjectInt32Param(j, sim.jointintparam_motor_enabled))

base = h["base"]
for t in (sim.scripttype_childscript, sim.scripttype_customizationscript):
    try:
        print("script type", t, "on UR5:", sim.getScript(t, base))
    except Exception as e:
        print("script type", t, "on UR5: none")

z.stop_simulation(client, sim)
client.setStepping(False)