def is_blank(string):
    if string and string.strip():
        # string is not None AND string is not empty or blank
        return False
    # string is None OR string is empty or blank
    return True
