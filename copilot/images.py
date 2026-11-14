"""Load problem images for the optional local vision model."""
import base64
import os


def path_for(problem, image_dir):
    filename = f"{problem['id']}_{problem['language']}.png"
    return os.path.join(image_dir, filename)


def load_b64(problem, image_dir):
    path = path_for(problem, image_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as image:
        return base64.b64encode(image.read()).decode("ascii")
