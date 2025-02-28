import os

images_path = '../images'

image_len = len(os.listdir(images_path))
current_len = image_len

for category in os.listdir("Images"):
    for url in os.listdir(f"Images/{category}"):
        old_path = os.path.join("Images", category, url)
        new_path = os.path.join(images_path, f'{current_len}-{category}.jpg')
        os.rename(old_path, new_path)

        current_len += 1