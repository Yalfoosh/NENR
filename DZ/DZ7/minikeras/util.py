def pad_string(string: str, maximum_length: int, prioritize_left_pad: bool = True, pad_with: str = " "):
    pad_length = maximum_length - len(string)

    if pad_length < 0:
        raise RuntimeError(f"String you're trying to pad (length {len(string)}) is too big for the size you're trying "
                           f"to pad it to ({maximum_length})!")

    left_pad = pad_length // 2

    if prioritize_left_pad and pad_length % 2 == 1:
        left_pad += 1

    right_pad = pad_length - left_pad

    return pad_with * left_pad + string + pad_with * right_pad

