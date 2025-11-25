def get_requirements_list(file_path: str = "requirements.txt") -> list:
    """
    Read requirements.txt and return list of package names

    Args:
        file_path: Path to requirements.txt file

    Returns:
        List of package names
    """
    with open(file_path, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return packages

print(get_requirements_list())