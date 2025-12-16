import os
from pathlib import Path
from dotenv import load_dotenv


def load_env_file(env_path: str = None):
    """
    Load environment variables from .env file

    Args:
        env_path: Path to .env file. If None, looks for .env in project root
    """
    if env_path is None:
        # Look for .env in project root (model directory)
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"

    if Path(env_path).exists():
        load_dotenv(env_path)
        return True
    return False


def get_env(key: str, default=None):
    """
    Get environment variable with optional default

    Args:
        key: Environment variable name
        default: Default value if not found
    """
    return os.environ.get(key, default)
