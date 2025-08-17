import importlib.util
import subprocess
import sys

# List of required packages
packages = [
    'pm4py',
    'scikit-learn',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'h5py',
    'scipy',
    'tensorflow',  # for deep learning models
    'torch',  # optional: PyTorch models
    'kmodes',
    'Levenshtein'
]

def install_package(package):
    """ Installs the package using pip. """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check each package
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    if spec:
        status = "✅ Installed"
    else:
        status = "❌ Not Installed"
        print(f"{pkg}: {status}")
        # Attempt to install the package if not installed
        print(f"Installing {pkg}...")
        install_package(pkg)
        print(f"{pkg} has been installed.\n")

# Verify installation status after installing missing packages
print("\nVerification of installed packages:")
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    status = "✅ Installed" if spec else "❌ Not Installed"
    print(f"{pkg}: {status}")
