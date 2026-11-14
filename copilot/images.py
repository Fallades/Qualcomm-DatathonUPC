"""Load and focus problem images for the optional local vision model.

The board intentionally has no image-processing dependency.  The small PNG
decoder below is enough for the dataset images and lets us find the
rightmost connected foreground component before sending it to the vision
model.
"""
import base64
import binascii
import io
import struct
import zlib
import os

MAX_MASK_SIDE = 320
MAX_OUTPUT_SIDE = 384
MASK_THRESHOLD = 245
PADDING = 12
DILATION_RADIUS = 2
MIN_COMPONENT_AREA = 3


def path_for(problem, image_dir):
    filename = f"{problem['id']}_{problem['language']}.png"
    return os.path.join(image_dir, filename)


def _png_pixels(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height, depth, color_type, _, _, interlace = struct.unpack(
        ">IIBBBBB", data[16:29])
    if depth != 8 or interlace != 0 or color_type not in (0, 2, 4, 6):
        return None
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    idat = bytearray()
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        chunk = data[offset + 8:offset + 8 + length]
        offset += length + 12
        if kind == b"IDAT":
            idat.extend(chunk)
        elif kind == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    rows = []
    previous = bytearray(stride)
    position = 0
    for _ in range(height):
        filter_type = raw[position]
        position += 1
        current = bytearray(raw[position:position + stride])
        position += stride
        for i in range(stride):
            left = current[i - channels] if i >= channels else 0
            above = previous[i]
            upper_left = previous[i - channels] if i >= channels else 0
            if filter_type == 1:
                current[i] = (current[i] + left) & 255
            elif filter_type == 2:
                current[i] = (current[i] + above) & 255
            elif filter_type == 3:
                current[i] = (current[i] + ((left + above) // 2)) & 255
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (abs(estimate - left), abs(estimate - above),
                             abs(estimate - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
                current[i] = (current[i] + predictor) & 255
            elif filter_type != 0:
                return None
        rows.append(bytes(current))
        previous = current
    return width, height, channels, rows


def _luminance(row, x, channels):
    if channels == 1:
        return row[x]
    if channels == 2:
        return row[x * 2]
    index = x * channels
    return (299 * row[index] + 587 * row[index + 1] + 114 * row[index + 2]) // 1000


def _rightmost_component(width, height, channels, rows):
    scale = max(1, (max(width, height) + MAX_MASK_SIDE - 1) // MAX_MASK_SIDE)
    small_width = (width + scale - 1) // scale
    small_height = (height + scale - 1) // scale
    mask = bytearray(small_width * small_height)
    for y in range(small_height):
        row = rows[min(y * scale, height - 1)]
        for x in range(small_width):
            if _luminance(row, min(x * scale, width - 1), channels) < MASK_THRESHOLD:
                mask[y * small_width + x] = 1

    seen = bytearray(len(mask))
    best = None
    for start in range(len(mask)):
        if not mask[start] or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        left = right = start % small_width
        top = bottom = start // small_width
        area = 0
        while stack:
            point = stack.pop()
            x, y = point % small_width, point // small_width
            area += 1
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                                   (x - 1, y - 1), (x + 1, y - 1),
                                   (x - 1, y + 1), (x + 1, y + 1)):
                if 0 <= next_x < small_width and 0 <= next_y < small_height:
                    next_point = next_y * small_width + next_x
                    if mask[next_point] and not seen[next_point]:
                        seen[next_point] = 1
                        stack.append(next_point)
        candidate = (right, area, left, top, bottom)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    _, _, left, top, bottom = best
    right = best[0]
    return (max(0, left * scale - PADDING),
            max(0, top * scale - PADDING),
            min(width, (right + 1) * scale + PADDING),
            min(height, (bottom + 1) * scale + PADDING))


def _encode_png(width, height, channels, rows):
    raw = b"".join(b"\0" + row for row in rows)

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", binascii.crc32(kind + payload) & 0xffffffff))

    header = struct.pack(">IIBBBBB", width, height, 8,
                         {1: 0, 2: 4, 3: 2, 4: 6}[channels], 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def _focus_png(data):
    decoded = _png_pixels(data)
    if decoded is None:
        return data
    width, height, channels, rows = decoded
    box = _rightmost_component(width, height, channels, rows)
    if box is None:
        return data
    left, top, right, bottom = box
    crop_width, crop_height = right - left, bottom - top
    scale = max(1, (max(crop_width, crop_height) + MAX_OUTPUT_SIDE - 1) // MAX_OUTPUT_SIDE)
    output_rows = [
        b"".join(row[(left + x) * channels:(left + x + 1) * channels]
                 for x in range(0, crop_width, scale))
        for y in range(0, crop_height, scale)
        for row in (rows[top + y],)
    ]
    return _encode_png(len(output_rows[0]) // channels, len(output_rows), channels, output_rows)


def _focus_jpeg(data):
    try:
        from PIL import Image
    except ImportError:
        return data
    with Image.open(io.BytesIO(data)) as source:
        gray = source.convert("L")
        scale = max(1, (max(gray.size) + MAX_MASK_SIDE - 1) // MAX_MASK_SIDE)
        small = gray.resize((max(1, gray.width // scale), max(1, gray.height // scale)))
        pixels = small.load()
        mask = bytearray(small.width * small.height)
        for y in range(small.height):
            for x in range(small.width):
                if pixels[x, y] < MASK_THRESHOLD:
                    mask[y * small.width + x] = 1
        for _ in range(DILATION_RADIUS):
            expanded = bytearray(mask)
            for y in range(small.height):
                for x in range(small.width):
                    if mask[y * small.width + x]:
                        continue
                    for next_y in range(max(0, y - 1), min(small.height, y + 2)):
                        for next_x in range(max(0, x - 1), min(small.width, x + 2)):
                            if mask[next_y * small.width + next_x]:
                                expanded[y * small.width + x] = 1
                                break
                        if expanded[y * small.width + x]:
                            break
            mask = expanded
        seen = bytearray(len(mask))
        best = None
        for start in range(len(seen)):
            if seen[start] or not mask[start]:
                continue
            stack = [start]
            seen[start] = 1
            left = right = start % small.width
            top = bottom = start // small.width
            area = 0
            while stack:
                point = stack.pop()
                x, y = point % small.width, point // small.width
                area += 1
                left, right = min(left, x), max(right, x)
                top, bottom = min(top, y), max(bottom, y)
                for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                                       (x - 1, y - 1), (x + 1, y - 1),
                                       (x - 1, y + 1), (x + 1, y + 1)):
                    if 0 <= next_x < small.width and 0 <= next_y < small.height:
                        next_point = next_y * small.width + next_x
                        if mask[next_point] and not seen[next_point]:
                            seen[next_point] = 1
                            stack.append(next_point)
            candidate = (right, area, left, top, bottom)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[1] < MIN_COMPONENT_AREA:
            return _resize_jpeg(data)
        _, _, left, top, bottom = best
        right = best[0]
        box = (max(0, left * scale - PADDING),
               max(0, top * scale - PADDING),
               min(gray.width, (right + 1) * scale + PADDING),
               min(gray.height, (bottom + 1) * scale + PADDING))
        crop = gray.crop(box)
        crop.thumbnail((MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE), Image.Resampling.BILINEAR)
        output = io.BytesIO()
        crop.save(output, format="PNG", optimize=False)
        return output.getvalue()


def _resize_jpeg(data):
    try:
        from PIL import Image
    except ImportError:
        return data
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("L")
        image.thumbnail((MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE), Image.Resampling.BILINEAR)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False)
        return output.getvalue()


def load_b64(problem, image_dir):
    path = path_for(problem, image_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as image:
        data = image.read()
    if data[:2] == b"\xff\xd8":
        # Without OCR options, the image may contain pictorial answer choices.
        # Keep the whole image so the vision model can inspect every choice.
        focused = (_resize_jpeg(data) if not problem.get("options")
                   else _focus_jpeg(data))
    else:
        focused = _focus_png(data)
    return base64.b64encode(focused).decode("ascii")
