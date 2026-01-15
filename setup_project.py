import os

PROJECT_NAME = "hybrid-qml-distribution-shift"

structure = {
    "data": ["cifar10", "cifar10c"],
    "models": ["__init__.py", "cnn.py", "mlp.py", "hybrid_qml.py"],
    "quantum": ["__init__.py", "circuits.py", "noise_models.py"],
    "shifts": ["__init__.py", "corruptions.py", "covariate.py"],
    "metrics": ["__init__.py", "calibration.py", "entropy.py"],
    "experiments": ["run_clean.py", "run_shifted.py", "run_noise.py"],
    "plots": [],
    "paper": ["figures", "draft.tex"]
}

def create_structure(base_path):
    for folder, items in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        for item in items:
            item_path = os.path.join(folder_path, item)
            if "." in item:  # file
                open(item_path, "a").close()
            else:  # subfolder
                os.makedirs(item_path, exist_ok=True)

def create_root_files(base_path):
    files = {
        "README.md": "# Hybrid Quantum–Classical Learning Under Distribution Shift\n",
        "requirements.txt": "",
        ".gitignore": "__pycache__/\n*.pyc\n.env\n"
    }

    for filename, content in files.items():
        with open(os.path.join(base_path, filename), "w") as f:
            f.write(content)

if __name__ == "__main__":
    os.makedirs(PROJECT_NAME, exist_ok=True)
    create_structure(PROJECT_NAME)
    create_root_files(PROJECT_NAME)

    print(f"Project '{PROJECT_NAME}' created successfully.")