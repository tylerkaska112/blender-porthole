# Compares two 8-bit PNG renders: mean colour of each, and the root-mean-square difference between them. python3 -I compare.py a.png b.png
import sys, zlib, struct
def load(p):
    d = open(p, 'rb').read(); i = 8; idat = b''; w = h = ct = bd = 0
    while i < len(d):
        n, t = struct.unpack('>I4s', d[i:i + 8]); c = d[i + 8:i + 8 + n]; i += 12 + n
        if t == b'IHDR': w, h, bd, ct = struct.unpack('>IIBB', c[:10])
        elif t == b'IDAT': idat += c
    assert bd == 8 and ct in (2, 6), (bd, ct)
    bpp = 3 if ct == 2 else 4; raw = zlib.decompress(idat); stride = w * bpp; out = []; prev = bytearray(stride); k = 0
    for y in range(h):
        f = raw[k]; row = bytearray(raw[k + 1:k + 1 + stride]); k += 1 + stride
        for x in range(stride):
            a = row[x - bpp] if x >= bpp else 0; b = prev[x]; c2 = prev[x - bpp] if x >= bpp else 0
            if f == 1: row[x] = (row[x] + a) & 255
            elif f == 2: row[x] = (row[x] + b) & 255
            elif f == 3: row[x] = (row[x] + ((a + b) >> 1)) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c2), abs(a - c2), abs(a + b - 2 * c2); pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c2); row[x] = (row[x] + pr) & 255
        out.append(row); prev = row
    return w, h, bpp, out
w, h, bpp, A = load(sys.argv[1]); w2, h2, bpp2, B = load(sys.argv[2]); assert (w, h) == (w2, h2), 'sizes differ'
def mean(img, bp): s = [0, 0, 0]; [s.__setitem__(c, s[c] + sum(r[c::bp][:w][i] for i in range(w))) for r in img for c in range(3)]; return [round(v / (w * h), 1) for v in s]
sq = 0; n = 0
for ra, rb in zip(A, B):
    for x in range(w):
        for c in range(3): d = ra[x * bpp + c] - rb[x * bpp2 + c]; sq += d * d; n += 1
print('mean colour', mean(A, bpp), mean(B, bpp2), 'RMS difference', round((sq / n) ** 0.5, 2))
