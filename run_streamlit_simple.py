import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_lgv_simple.py"], check=True)


if __name__ == "__main__":
    main()
