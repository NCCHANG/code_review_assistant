import kagglehub

# Download latest version
path = kagglehub.dataset_download("shamimhasan8/python-code-bug-and-fix-pairs",output_dir='./datasets')

print("Path to dataset files:", path)