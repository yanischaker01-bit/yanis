import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_lgv_pro.py"], check=True)


if __name__ == "__main__":
    main()
