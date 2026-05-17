import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import ensure_project_directories

def main():
    ensure_project_directories()
    print("Project directories successfully initialized.")

if __name__ == "__main__":
    main()
