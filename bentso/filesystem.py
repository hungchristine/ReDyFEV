import appdirs
import hashlib
import os


DEFAULT_DATA_DIR = appdirs.user_data_dir("bentso", "bonsai")


def load_token():
    cwd = os.getcwd()
    token = os.environ.get('ENTSOE_API_TOKEN')
    # TODO: Consider https://github.com/theskumar/python-dotenv
    if not token and "entsoe_api_token.txt" in os.listdir(cwd):
        token = open("entsoe_api_token.txt").readlines()[0].strip()
    if not token:
        raise ValueError("Can't find ENTSO-E token")
    return token


def create_dir(dirpath):
    """Create directory tree to `dirpath`; ignore if already exists"""
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)


def sha256(filepath, blocksize=65536):
    """Generate SHA 256 hash for file at `filepath`"""
    hasher = hashlib.sha256()
    fo = open(filepath, 'rb')
    buf = fo.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fo.read(blocksize)
    return hasher.hexdigest()
