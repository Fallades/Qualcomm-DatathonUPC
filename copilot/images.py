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
FULL_IMAGE_MAX_SIDE = 384
MASK_THRESHOLD = 245
PADDING = 12
DILATION_RADIUS = 4
MIN_COMPONENT_AREA = 5
OPTION_BAND_FRACTION = 0.35
OPTION_SEARCH_TOP_FRACTION = 0.45


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
    """Return grayscale luminance, compositing transparency over white."""
    i = x * channels

    if channels == 1:
        return row[i]

    if channels == 2:
        gray, alpha = row[i], row[i + 1]
        return (gray * alpha + 255 * (255 - alpha)) // 255

    gray = (
        299 * row[i] +
        587 * row[i + 1] +
        114 * row[i + 2]
    ) // 1000

    if channels == 4:
        alpha = row[i + 3]
        gray = (gray * alpha + 255 * (255 - alpha)) // 255

    return gray


def _dilate_mask(mask, width, height, radius):
    """Expand foreground regions using a square morphological kernel."""
    if radius <= 0:
        return mask

    expanded = bytearray(mask)

    for y in range(height):
        for x in range(width):
            if not mask[y * width + x]:
                continue

            for dy in range(-radius, radius + 1):
                ny = y + dy
                if not 0 <= ny < height:
                    continue

                for dx in range(-radius, radius + 1):
                    nx = x + dx
                    if 0 <= nx < width:
                        expanded[ny * width + nx] = 1

    return expanded


def _rightmost_component(width, height, channels, rows, y_limit=None):
    """Find the rightmost significant foreground component."""

    scale = max(
        1,
        (max(width, height) + MAX_MASK_SIDE - 1) // MAX_MASK_SIDE
    )

    small_width = (width + scale - 1) // scale
    search_height = height if y_limit is None else min(height, y_limit)
    small_height = (search_height + scale - 1) // scale

    # 1. Build binary foreground mask.
    mask = bytearray(small_width * small_height)

    for y in range(small_height):
        row = rows[min(y * scale, height - 1)]

        for x in range(small_width):
            luminance = _luminance(
                row,
                min(x * scale, width - 1),
                channels
            )

            if luminance < MASK_THRESHOLD:
                mask[y * small_width + x] = 1

    # 2. Dilate to connect nearby strokes.
    mask = _dilate_mask(
        mask, small_width, small_height, DILATION_RADIUS
    )

    # 3. Find connected components using 8-connectivity.
    seen = bytearray(len(mask))
    components = []

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
            x = point % small_width
            y = point // small_width

            area += 1
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < small_width and
                            0 <= ny < small_height):
                        continue

                    neighbor = ny * small_width + nx

                    if mask[neighbor] and not seen[neighbor]:
                        seen[neighbor] = 1
                        stack.append(neighbor)

        if area >= MIN_COMPONENT_AREA:
            components.append((right, area, left, top, bottom))

    if not components:
        return None

    # 4. Prefer the component extending furthest right.
    # Area breaks ties, reducing sensitivity to small noise.
    right, area, left, top, bottom = max(components)

    # 5. Convert bounding box to original image coordinates.
    return (
        max(0, left * scale - PADDING),
        max(0, top * scale - PADDING),
        min(width, (right + 1) * scale + PADDING),
        min(height, (bottom + 1) * scale + PADDING)
    )


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
    crop_width = right - left
    crop_height = bottom - top

    scale = max(
        1,
        (max(crop_width, crop_height) + MAX_OUTPUT_SIDE - 1)
        // MAX_OUTPUT_SIDE
    )

    output_rows = [
        b"".join(
            row[(left + x) * channels:(left + x + 1) * channels]
            for x in range(0, crop_width, scale)
        )
        for y in range(0, crop_height, scale)
        for row in (rows[top + y],)
    ]

    result = _encode_png(
        len(output_rows[0]) // channels,
        len(output_rows),
        channels,
        output_rows
    )

    return result



def _focus_jpeg(data, include_options=False):
    try:
        from PIL import Image
    except ImportError:
        return data

    with Image.open(io.BytesIO(data)) as source:
        gray = source.convert("L")
        width, height = gray.size
        pixels = gray.load()

        # Reuse exactly the same segmentation logic as PNG.
        # Represent grayscale pixels as one-channel rows.
        rows = [
            bytes(pixels[x, y] for x in range(width))
            for y in range(height)
        ]

        box = _rightmost_component(width, height, 1, rows)

        if box is None:
            return _resize_jpeg(data)

        crop = gray.crop(box)
        if include_options:
            options_top = int(gray.height * (1.0 - OPTION_BAND_FRACTION))
            options = gray.crop((0, options_top, gray.width, gray.height))
            crop.thumbnail((MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE), Image.Resampling.BILINEAR)
            options.thumbnail((FULL_IMAGE_MAX_SIDE, FULL_IMAGE_MAX_SIDE), Image.Resampling.BILINEAR)
            width = max(crop.width, options.width)
            combined = Image.new("L", (width, crop.height + options.height), 255)
            combined.paste(crop, ((width - crop.width) // 2, 0))
            combined.paste(options, ((width - options.width) // 2, crop.height))
            crop = combined
        crop.thumbnail(
            ((FULL_IMAGE_MAX_SIDE if include_options else MAX_OUTPUT_SIDE),
             (FULL_IMAGE_MAX_SIDE if include_options else MAX_OUTPUT_SIDE)),
            Image.Resampling.BILINEAR
        )

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
        image.thumbnail((FULL_IMAGE_MAX_SIDE, FULL_IMAGE_MAX_SIDE), Image.Resampling.BILINEAR)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False)
        return output.getvalue()


def _options_jpeg(data):
    try:
        from PIL import Image
    except ImportError:
        return data
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("L")
        options_top = int(image.height * (1.0 - OPTION_BAND_FRACTION))
        box = (0, options_top, image.width, image.height)
        options = image.crop(box)
        options.thumbnail((FULL_IMAGE_MAX_SIDE, FULL_IMAGE_MAX_SIDE),
                          Image.Resampling.BILINEAR)
        output = io.BytesIO()
        options.save(output, format="PNG", optimize=False)
        return output.getvalue()


def load_b64(problem, image_dir, save_dir=None):
    path = path_for(problem, image_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as image:
        data = image.read()
    if data[:2] == b"\xff\xd8":
        focused = ([data]
                   if not problem.get("options") else [_focus_jpeg(data)])
    else:
        focused = ([data]
                   if not problem.get("options") else [_focus_png(data)])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem, extension = os.path.splitext(os.path.basename(path))
        labels = ("diagram", "options") if len(focused) == 2 else ("diagram",)
        for label, image_data in zip(labels, focused):
            output_path = os.path.join(save_dir, f"{stem}_{label}{extension}")
            with open(output_path, "wb") as output:
                output.write(image_data)
    return [base64.b64encode(image).decode("ascii") for image in focused]
