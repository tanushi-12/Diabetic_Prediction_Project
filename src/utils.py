import os


def create_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)


def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)