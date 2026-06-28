import os
from dotenv import load_dotenv


def load_env():
    env_file = os.getenv("ENV_FILE", ".env")
    print(f"Loading environment from: {env_file}")
    load_dotenv(dotenv_path=env_file, override=True)
