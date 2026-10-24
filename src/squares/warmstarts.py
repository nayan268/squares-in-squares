"""Extend saved packings with non-overlapping unit-square border strips."""
import math
from .geometry import Square,validate


def add_border(packing,clearance=1e-7):
    side=validate(packing)
    points=[point for square in packing for point in square.corners()]
    xmin=min(p[0] for p in points)
    ymin=min(p[1] for p in points)
    result=[Square(s.x-xmin,s.y-ymin,s.angle) for s in packing]
    count=math.floor(side)
    # New top and right strips are separated from the old bounding square.
    for j in range(count):
        result.append(Square(j+0.5,side+0.5+clearance))
    for j in range(count+1):
        result.append(Square(side+0.5+clearance,j+0.5))
    validate(result)
    return result


def extended_starts(saved,max_n=1000):
    result={}
    for packing in saved.values():
        current=list(packing)
        while len(current)<=max_n:
            n=len(current)
            side=validate(current,n)
            if n not in result or side<validate(result[n],n):
                result[n]=current
            current=add_border(current)
    return result
