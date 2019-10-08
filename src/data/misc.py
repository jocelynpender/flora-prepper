def is_blank(string): # https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty
    if string and string.strip():
        # string is not None AND string is not empty or blank
        return False
    # string is None OR string is empty or blank
    return True
