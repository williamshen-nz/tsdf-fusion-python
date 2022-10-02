import trimesh


def view(mesh_file: str) -> None:
    mesh = trimesh.load(mesh_file)
    mesh.show()


if __name__ == "__main__":
    view("mesh.ply")
