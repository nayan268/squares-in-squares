"""Published small-case constructions used as optimizer seeds.

The n=11 construction follows the exact formulas in the Squares Project's
Trump witness. The n=18 and n=19 centers are the project's CC BY 4.0
known-best witness data, attributed to Joshua Levy, the Squares Project:
https://github.com/jlevy/squares

Original constructions include Trump (1979), Hamalainen (1980), Wainwright
(1979), Bidwell (1998), and Schadt (2025).
A tiny uniform expansion of the centers supplies numerical clearance.
"""
from decimal import Decimal, getcontext
import math

from .geometry import Square, validate


_DATA = {
    17: [
        (0.5,0.5,0),(1.5,0.5,0),(0.5,1.5,0),
        (0.5,4.175530093604551,0),(4.175530093604551,0.5,0),
        (3.175530093604551,0.5,0),(4.175530093604551,1.5,0),
        (4.175530093604551,4.175530093604551,0),
        (0.7042021556575175,2.7042021556575175,39.8049589797678),
        (1.4360434939982247,3.388043433263363,39.8049589797678),
        (1.556737163170879,2.1129358718403517,39.8049589797678),
        (2.2885785015115863,2.7967771494461973,39.8049589797678),
        (2.295286446181523,1.4266835476667392,39.8049589797678),
        (3.0271277845222304,2.110524825272585,39.8049589797678),
        (2.347324826510285,4.175530093604551,0),
        (3.245073712656966,3.3724938614938536,53.37621361655341),
        (4.175530093604551,2.613460132512564,0),
    ],
    18: [
        (0.5,0.5,0),(0.5,1.5,0),(0.5,2.5,0),(1.5,0.5,0),
        (0.5,4.322875655532295,0),
        (3.4700161985907583,1.3892972847076845,24.295188945364572),
        (3.058578370824611,2.300735112473832,24.295188945364572),
        (2.566421629175389,3.175735112473832,24.295188945364572),
        (2.1549838014092417,4.08717294023998,24.295188945364572),
        (4.322875655532295,4.322875655532295,0),
        (4.322875655532295,3.3228756555322954,0),
        (4.322875655532295,2.3228756555322954,0),
        (3.3228756555322954,4.322875655532295,0),
        (4.322875655532295,0.5,0),
        (1.3528594569415369,3.433578370824611,24.295188945364572),
        (1.7642972847076845,2.522140543058463,24.295188945364572),
        (2.256454026356906,1.6471405430584631,24.295188945364572),
        (2.6678918541230536,0.7357027152923155,24.295188945364572),
    ],
    19: [
        (0.5,4.385618083164127,0),(1.5,4.385618083164127,0),
        (0.5,3.3856180831641267,0),(4.385618083164127,0.5,0),
        (3.3856180831641267,0.5,0),(2.3856180831641267,0.5,0),
        (4.385618083164127,1.5,0),(3.3856180831641267,1.5,0),
        (4.385618083164127,2.5,0),(4.385618083164127,4.385618083164127,0),
        (3.6213203435596424,3.4428090415820636,45),
        (2.914213562373095,4.149915822768611,45),
        (2.9428090415820636,2.7071067811865475,45),
        (2.2357022603955157,3.414213562373095,45),
        (0.5,0.5,0),(1.4428090415820634,1.264297739604484,45),
        (0.7357022603955159,1.9714045207910318,45),
        (2.1785113019775793,1.9428090415820634,45),
        (1.4714045207910318,2.649915822768611,45),
    ],
    29: [
        (0.5,5.43383346267693,0),(1.5,5.43383346267693,0),
        (0.5,4.43383346267693,0),(5.43383346267693,5.43383346267693,0),
        (4.43383346267693,5.43383346267693,0),
        (3.433833462676929,5.43383346267693,0),
        (5.43383346267693,4.43383346267693,0),
        (5.43383346267693,3.433833462676929,0),
        (0.5,0.5,0),(1.5,0.5,0),(5.43383346267693,0.5,0),
        (4.43383346267693,0.5,0),(3.433833462676929,0.5,0),
        (5.43383346267693,1.5,0),(4.43383346267693,1.5,0),
        (2.4088726985517943,4.628659574271335,-25.258655308351404),
        (3.3475773535780506,4.269528495552081,-24.962587989437722),
        (4.269528495552081,3.8805142820266587,-24.962587989437722),
        (2.5262168697023952,1.334967158655357,-20.80012676269961),
        (1.5913919789567146,1.6900761892957028,-20.80012676269961),
        (0.6565670882110337,2.0451852199360485,-20.80012676269961),
        (4.5062540007573775,2.627247092891466,17.506268475732366),
        (3.4232587159292494,2.063929712272198,-20.80012676269961),
        (2.4884338251835683,2.419038742912544,-20.80012676269961),
        (1.5536089344378876,2.7741477735528894,-20.80012676269961),
        (0.6449669606930133,3.1981833603006518,-20.80012676269961),
        (1.543385771683897,3.9221995880176532,-24.308358401346908),
        (2.479040566219762,3.49232598741976,-20.80012676269961),
        (3.4138654569654427,3.1372169567794144,-20.80012676269961),
    ],
}


def _expanded(squares, factor=1+2e-8):
    cx=sum(s.x for s in squares)/len(squares)
    cy=sum(s.y for s in squares)/len(squares)
    result=[Square(cx+(s.x-cx)*factor, cy+(s.y-cy)*factor, s.angle)
            for s in squares]
    validate(result, len(result))
    return result


def _poly(value):
    coefficients=(5,-10,-2,14,12,-6,2,2,-1)
    result=Decimal(0)
    for coefficient in coefficients:
        result=result*value+Decimal(coefficient)
    return result


def trump_eleven():
    getcontext().prec=70
    low, high=Decimal("0.36"), Decimal("0.37")
    low_sign=_poly(low)>0
    for _ in range(240):
        middle=(low+high)/2
        if (_poly(middle)>0)==low_sign:
            low=middle
        else:
            high=middle
    u=(low+high)/2
    one=Decimal(1)
    cos_a=(one-u*u)/(one+u*u)
    sin_a=2*u/(one+u*u)
    side=(6*u+4)/(one+2*u-u*u)
    r1=one-(side-3)*cos_a
    u1=((one+r1)*cos_a-one)/sin_a
    v1=cos_a-sin_a
    v2=(side-one)/sin_a-r1-(3+u1)*(cos_a/sin_a)
    x0=one+2/cos_a-(side-2)*(sin_a/cos_a)

    squares=[
        Square(Decimal("0.5"),Decimal("0.5"),0),
        Square(side-Decimal("0.5"),Decimal("0.5"),0),
        Square(x0+Decimal("0.5"),side-Decimal("0.5"),0),
        Square(Decimal("0.5"),side-Decimal("0.5"),0),
        Square(Decimal("1.5"),side-Decimal("0.5"),0),
        Square(Decimal("0.5"),side-Decimal("1.5"),0),
    ]
    angle=math.atan2(float(sin_a),float(cos_a))
    for ox, oy in ((0,0),(u1,-1),(1,v1),(u1+1,v1-1),(u1+2,-v2)):
        px=Decimal(ox)+Decimal("0.5")
        py=Decimal(oy)+Decimal("0.5")-r1
        x=one+cos_a*px-sin_a*py
        y=one+sin_a*px+cos_a*py
        squares.append(Square(float(x),float(y),angle))
    squares=[Square(float(s.x),float(s.y),float(s.angle)) for s in squares]
    return _expanded(squares)


def published_seed(n):
    if n==11:
        return trump_eleven()
    if n not in _DATA:
        raise KeyError(n)
    squares=[Square(x,y,math.radians(angle)) for x,y,angle in _DATA[n]]
    return _expanded(squares)


def available():
    return (11,17,18,19,29)


def propagated_seed(n, clearance=1e-7):
    """Best exact border-family descendant of a published small seed."""
    from .warmstarts import add_border
    candidates=[]
    for base in available():
        current=published_seed(base)
        while len(current)<n:
            current=add_border(current,clearance)
        if len(current)==n:
            candidates.append(current)
    if not candidates:
        raise KeyError(n)
    return min(candidates,key=validate)


def propagated_seeds(max_n=1000, clearance=1e-7):
    """Generate every distinct published-seed border descendant once."""
    from .warmstarts import add_border
    if not 1 <= max_n <= 1000:
        raise ValueError("max_n must be from 1 through 1000")
    result={}
    for base in available():
        current=published_seed(base)
        while len(current)<=max_n:
            n=len(current)
            if n not in result or validate(current,n)<validate(result[n],n):
                result[n]=current
            current=add_border(current,clearance)
    return result
