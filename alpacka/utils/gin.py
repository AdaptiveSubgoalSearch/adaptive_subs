"""Utilities for Gin configs."""


def extract_bindings(config_str):
    """Extracts bindings from a Gin config string.

    Args:
        config_str (str): Config string to parse.

    Returns:
        List of (name, value) pairs of the extracted Gin bindings.
    """
    # Really crude parsing of gin configs.
    # Remove line breaks preceded by '\'.
    config_str = config_str.replace('\\\n', '')
    # Remove line breaks inside parentheses. Those are followed by indents.
    config_str = config_str.replace('\n    ', '')
    # Indents starting with parentheses are 3-space.
    config_str = config_str.replace('\n   ', '')
    # Lines containing ' = ' are treated as bindings, everything else is
    # ignored.
    sep = ' = '

    bindings = []
    for line in config_str.split('\n'):
        line = line.strip()
        if sep in line:
            chunks = line.split(sep)
            name = chunks[0].strip()
            value = sep.join(chunks[1:]).strip()
            bindings.append((name, value))
    return bindings
