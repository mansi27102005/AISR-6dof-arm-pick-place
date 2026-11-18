import zmq_client as z

client, sim = z.connect()
h = z._handles(sim)
s = sim.getScript(sim.scripttype_childscript, h["base"])
print("script handle:", s)
print(sim.getScriptStringParam(s, sim.scriptstringparam_text))
client.setStepping(False)