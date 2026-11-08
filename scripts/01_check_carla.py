import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
client_version = client.get_client_version()
server_version = client.get_server_version()
print("Client:", client_version)
print("Server:", server_version)
if client_version != server_version:
    raise RuntimeError("CARLA client/server version mismatch")
world = client.get_world()
print("Map:", world.get_map().name)
print("Actor count:", len(world.get_actors()))
print("CARLA connection OK")
