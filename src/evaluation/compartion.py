from vae_visuals import plot_reconstructions_from_paths

model_paths = [
    "vae_minecraft_small.pth",
    "vae_minecraft_medium.pth",
    "vae_minecraft_large.pth",
]
test_images = [

    "src/metrics/example.png",
    "src/metrics/example2.png",
    "src/metrics/example3.png",
]
plot_reconstructions_from_paths(model_paths, test_images)