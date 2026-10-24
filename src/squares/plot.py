"""Packing diagrams with overlap highlighting (SVG, optionally PNG via Pillow)."""
import html
import math
from pathlib import Path
from .geometry import container_side, overlaps


def plot_packings(panels, output, title="Squares in Squares"):
    """panels is a sequence of (label, packing). Never overwrite a diagram."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    if not panels or any(not packing for _, packing in panels):
        raise ValueError("panels must contain nonempty packings")
    cols = min(3,len(panels))
    rows = math.ceil(len(panels)/cols)
    width, height = cols*420, 90+rows*460
    colors = ["#5b8ff9","#61c4a4","#f2bd56","#a187db","#f18e79","#79bbd7"]
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#f6f8fc"/>',
                f'<text x="30" y="42" font-family="sans-serif" font-size="25" font-weight="bold" fill="#17263c">{html.escape(title)}</text>',
                '<text x="30" y="68" font-family="sans-serif" font-size="13" fill="#536279">Unit squares | red indicates a detected overlap</text>']
    draw_commands = []
    for panel,(label,packing) in enumerate(panels):
        points = [p for s in packing for p in s.corners()]
        xmin,ymin = min(p[0] for p in points),min(p[1] for p in points)
        side = container_side(packing)
        scale = 350/side
        ox,oy = 35+(panel%cols)*420, 90+(panel//cols)*460
        bad = set()
        for i,a in enumerate(packing):
            for j in range(i):
                if overlaps(a,packing[j]):
                    bad.update((i,j))
        heading = f"{label} | n={len(packing)}"
        subtitle = f"L = {side:.9f} | " + ("OVERLAP" if bad else "local check passed")
        elements += [f'<text x="{ox}" y="{oy+18}" font-family="sans-serif" font-size="17" font-weight="bold" fill="#17263c">{html.escape(heading)}</text>',
                     f'<text x="{ox}" y="{oy+39}" font-family="sans-serif" font-size="13" fill="#536279">{subtitle}</text>']
        draw_commands += [("text",(ox,oy+2),heading,17,"#17263c"),
                          ("text",(ox,oy+25),subtitle,13,"#536279")]
        top = oy+58
        elements.append(f'<rect x="{ox}" y="{top}" width="350" height="350" fill="white" stroke="#17263c" stroke-width="2"/>')
        draw_commands.append(("rectangle",(ox,top,ox+350,top+350),"white",2,"#17263c"))
        for i,s in enumerate(packing):
            corners = [(ox+(x-xmin)*scale,top+350-(y-ymin)*scale) for x,y in s.corners()]
            fill = "#ee6a70" if i in bad else colors[i%len(colors)]
            outline = "#b6202a" if i in bad else "#294461"
            coords = " ".join(f"{x:.5f},{y:.5f}" for x,y in corners)
            elements.append(f'<polygon points="{coords}" fill="{fill}" fill-opacity=".75" stroke="{outline}" stroke-width="1.5"/>')
            draw_commands.append(("polygon",corners,fill,2,outline))
            cx,cy = ox+(s.x-xmin)*scale,top+350-(s.y-ymin)*scale
            elements.append(f'<text x="{cx:.5f}" y="{cy+5:.5f}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#17263c">{i}</text>')
            draw_commands.append(("label",(cx,cy),str(i),14,"#17263c"))
    elements.append("</svg>")
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()==".svg":
        path.write_text("\n".join(elements),encoding="utf-8")
    elif path.suffix.lower()==".png":
        try:
            from PIL import Image,ImageDraw,ImageFont
        except ImportError as error:
            raise RuntimeError("Install PNG support: pip install -e '.[plot]'") from error
        image = Image.new("RGB",(width*2,height*2),"#f6f8fc")
        draw = ImageDraw.Draw(image)
        def font(size):
            for name in ("C:/Windows/Fonts/segoeui.ttf","DejaVuSans.ttf"):
                try:
                    return ImageFont.truetype(name,size*2)
                except OSError:
                    pass
            return ImageFont.load_default()
        draw.text((60,30),title,font=font(25),fill="#17263c")
        draw.text((60,106),"Unit squares | red indicates a detected overlap",
                  font=font(13),fill="#536279")
        for kind,coords,text,size,color in draw_commands:
            if kind=="polygon":
                draw.polygon([(round(x*2),round(y*2)) for x,y in coords],fill=text)
                draw.line([(round(x*2),round(y*2)) for x,y in coords+[coords[0]]],fill=color,width=size*2)
            elif kind=="rectangle":
                draw.rectangle(tuple(round(v*2) for v in coords),fill=text,outline=color,width=size*2)
            else:
                point=tuple(round(v*2) for v in coords)
                draw.text(point,text,font=font(size),fill=color,anchor="mm" if kind=="label" else "lt")
        image.save(path)
    else:
        raise ValueError("plot output must end in .svg or .png")
    return path
