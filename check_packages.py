import importlib.util

packages = ["pm4py", "pandas", "matplotlib", "sklearn", "torch", "Levenshtein"]

for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    status = "✅ Installed" if spec else "❌ Not Installed"
    print(f"{pkg}: {status}")
