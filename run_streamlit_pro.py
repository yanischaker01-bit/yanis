import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Lance le tableau de bord LGV SEA avec Streamlit."""

    app_path = Path(__file__).resolve().parent / "streamlit_lgv_pro.py"

    if not app_path.exists():
        raise FileNotFoundError(
            f"Le fichier Streamlit est introuvable : {app_path}"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
