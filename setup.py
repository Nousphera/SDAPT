"""Install SDAPT directly from this source checkout."""
from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
requirements = [line.strip() for line in (root / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith(("#", "-e"))]
setup(
    name="sdapt", version="0.2.0",
    description="Domain Adaptive Pretraining in Split Learning for Vision Foundation Models",
    python_requires=">=3.12,<3.14",
    package_dir={"": "src"}, packages=find_packages("src"),
    install_requires=requirements,
)
