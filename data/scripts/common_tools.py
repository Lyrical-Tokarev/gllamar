import numpy as np

def make_square(bbox, height, width):
    """Extends bounding box to square if possible and returns new shape
    """
    if len(bbox) == 2:
        (x, y), (u, v) = bbox
        return_points = True
    else:
        x, y, u, v = bbox
        return_points = False
    if x > u:
        x, u = u, x
    if y > v:
        y, v = v, y
    if u >= 0 and u <= 1 and v >=0 and v <= 1:
        # have relative coorinates, translate them to pixel coordinates
        x = int(np.floor(x*width))
        y = int(np.floor(y*height))
        u = int(np.ceil(u*width))
        v = int(np.ceil(v*height))
    dx = u - x
    dy = v - y
    if dx == dy:
        if return_points:
            return (x, y), (u, v)
        return x, y, u, v
    size = max(dx, dy)
    #print(dx, dy)
    pad = int(np.abs(dx - dy) / 2)
    #print(pad)
    if size == dx:
        # extend dy
        y = max(y - pad, 0)
        v = min(y + size, height - 1)
    else:
        # extend dx
        x = max(x - pad, 0)
        u = min(x + size, width - 1)
    if return_points:
        return (x, y), (u, v)
    return x, y, u, v
