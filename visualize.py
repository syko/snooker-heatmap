import math
import heatmap
from PIL import Image

def get_hm_color(v, min_v, max_v):
    """Return the color value for the heatmap pixel with value v, when
    the minimum and maximum values for the whole heatmap are min_v and max_v.
    min_v <= v <= max_v assumed!"""
    coef = (float(v)-float(min_v)) / (float(max_v)-float(min_v))
    #return (int(0xff*coef) << 24) + (int(0xff*coef) << 16) + (int(0xff*coef) << 8) + int(0xff*coef)
    #return ( int(0xff*coef), int(0xff*coef), int(0xff*coef), int(0xff*coef) ) # black to white
    return (
        int(0xff*( min(1,coef*2) )),
        int(0xff*( 1-abs(0.5-coef)*2 )),
        int(0xff*( max(0,1-coef-0.5)*2 )),
        int(0xff*( min(0.8,0.2+coef*2) ))
    )

def make_heatmap(values, w, h, r=20):
    img = Image.new("RGBA", (w,h), (0,0,0,0))
    imgdata = img.load()
    density = []
    max_d = 0
    min_d = 999999
    for y in range(h):
        density_row = []
        for x in range(w):
            v = 0
            for p in values:
                if abs(x-p[0]) <= r and abs(y-p[1]) <= r:
                    v += max(0, r - math.sqrt( (x-p[0])**2 + (y-p[1])**2 ))
            max_d = max(max_d, v)
            min_d = min(min_d, v)
            density_row.append(v)
        density.append(density_row)
    for y in range(h):
        for x in range(w):
            #img.putpixel((x,y), get_hm_color(density[y][x], min_d, max_d))
            imgdata[x,y] = get_hm_color(density[y][x], min_d, max_d)
    return img

if __name__ == "__main__":

    pts = []

    for fp in [ open('confirmed_cueballs%s.txt' % fn, 'r') for fn in ['_1', '_2'] ]:
        for line in fp:
            if len(line.split(' ')):
                pts.append((
                    int(line.split(' ')[1]),
                    int(line.split(' ')[2])
                ))

    print "Processing %d points..." % len(pts)

    img = make_heatmap(pts, 704, 400, 30)
    img.save('heatmap.png')
