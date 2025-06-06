import math

def euclideanDistance(point1, point2):
    """
    Calculates Euclidean distance between two points (x, y).
    Suitable for measuring how far the shuttlecock moved between frames.
    """
    if None in point1 or None in point2:
        return float('inf')  # Invalid points should not be compared
    return math.dist(point1, point2)

def withinCircle(center, radius, point):
    """
    Checks if a detected shuttlecock location is within a certain radius.
    In badminton, reduce the radius because shuttlecock moves in tighter arcs.
    """
    return euclideanDistance(center, point) < radius  # '>' changed to '<' to be logically correct

def closestPoint(prevCenter, currCenter, prevPoint, currPoint):
    """
    Chooses the point that is closer to its respective frame center.
    This helps filter out jittery or false shuttlecock detections.
    In badminton, rapid changes are normal, so we might need to be more lenient.
    """
    prevDist = euclideanDistance(prevCenter, prevPoint)
    currDist = euclideanDistance(currCenter, currPoint)

    return prevDist <= currDist  # Returns True if previous position is more stable
"""Use tighter radii (e.g., 20–30 pixels instead of 50–100 for tennis).

Add direction check (to allow sharp angle shifts common in badminton).

Filter very small movements (shuttlecock sometimes hovers — not meaningful).

Time-based smoothing if jitter becomes an issue."""

