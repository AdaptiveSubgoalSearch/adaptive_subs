def readable_num(value):
    if 0 < value < 1e-2:
        return '{:>6.1e}'.format(value)
    else:
        return '{:>6.3f}'.format(value)
