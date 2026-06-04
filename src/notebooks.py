import json
from pathlib import Path

# Path to notebook directory
NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"

def list_notebooks() -> list[dict]:
    """Scan directory for notebooks and return metadata"""
    notebooks = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        stat = path.stat()
        notebooks.append({
            "name": path.stem,
            "filename": path.name,
            "size_bytes": stat.st_size  
        })
    return notebooks

def get_notebook(name: str) -> dict:
    """Safely load a single notebook by name."""
    filename = name if name.endswith(".ipynb") else f"{name}.ipynb"
    path = (NOTEBOOK_DIR / filename).resolve()
    
    # Ensure file exists
    if not path.exists():
        raise FileNotFoundError(f"Notebook '{name}' not found")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

