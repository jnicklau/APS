from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, LogNorm
MATPLOTLIB_INSTALLED = True

def cmap_discrete(cmap_list):
    """
    Can be used to create a discrete colormap.
    INPUT:
        - cmap_list (list) - list of tuples, where each tuple represents one range. Each tuple has the form of 
            ((from, to), color) .
    OUTPUT:
        - cmap - matplotlib colormap
        - norm - matplotlib norm object
    """
    if not MATPLOTLIB_INSTALLED:
        raise UserWarning("install matplotlib to use this function")
    cmap_colors = []
    boundaries = []
    last_upper = None
    for (lower, upper), color in cmap_list:
        if last_upper is not None and lower != last_upper:
            raise ValueError("Ranges for colormap must be continuous")
        cmap_colors.append(color)
        boundaries.append(lower)
        last_upper = upper
    boundaries.append(upper)
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm
