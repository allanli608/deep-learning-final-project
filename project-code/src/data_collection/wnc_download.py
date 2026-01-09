import kagglehub

# Download latest version
path = kagglehub.dataset_download("chandiragunatilleke/wiki-neutrality-corpus")

print("Path to dataset files:", path)
