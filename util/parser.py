def load_classes(path):
    with open(path, "r") as f:
        names = f.read().split("\n")
    return names