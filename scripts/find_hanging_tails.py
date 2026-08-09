import re

BODY_X0, BODY_W = 70.9, 453.5
attr = lambda s, k: float(re.search(k + r'="([\d.]+)"', s).group(1))
pageno, found = 0, 0
block_lines, in_block, block_x0 = [], False, 0.0

def flush():
    global found
    if len(block_lines) < 2 or not (60 < block_x0 < 95):
        return
    if max(w for w, _, _ in block_lines) < 350:
        return
    w, xmax, words = block_lines[-1]
    fill = (xmax - BODY_X0) / BODY_W
    if fill < 0.20:
        first = " ".join(block_lines[0][2][:6])
        print(f"p{pageno:3d} {fill*100:5.1f}%  tail={' '.join(words)!r}  start={first!r}")
        found += 1

with open("main_bbox.xml", errors="replace") as f:
    cur_words, cur_line = [], None
    for raw in f:
        s = raw.strip()
        if s.startswith("<page"):
            pageno += 1
        elif s.startswith("<block"):
            in_block, block_lines = True, []
            block_x0 = attr(s, "xMin")
        elif s.startswith("</block"):
            flush(); in_block = False
        elif s.startswith("<line") and in_block:
            cur_line = (attr(s, "xMin"), attr(s, "xMax")); cur_words = []
        elif s.startswith("<word") and cur_line:
            m = re.search(r">([^<]*)</word>", s)
            cur_words.append(m.group(1) if m else "")
        elif s.startswith("</line") and in_block and cur_line:
            block_lines.append((cur_line[1] - cur_line[0], cur_line[1], cur_words))
            cur_line = None
print(f"-- {found} hanging tails under 20% fill")
