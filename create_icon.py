import os
from PIL import Image, ImageDraw, ImageFilter

def generate_module_icon(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    size = (128, 128)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient background rounded rectangle
    r = 24
    draw.rounded_rectangle([(4, 4), (124, 124)], radius=r, fill=(24, 119, 242, 255))
    
    # Inner subtle glow
    draw.rounded_rectangle([(8, 8), (120, 120)], radius=r-2, outline=(255, 255, 255, 60), width=2)

    # Camera body
    draw.rounded_rectangle([(32, 46), (96, 96)], radius=10, fill=(255, 255, 255, 240))
    # Camera top notch
    draw.rounded_rectangle([(48, 38), (80, 48)], radius=4, fill=(255, 255, 255, 240))
    
    # Lens outer circle
    draw.ellipse([(48, 54), (80, 86)], fill=(24, 119, 242, 255))
    # Lens glass inner
    draw.ellipse([(54, 60), (74, 80)], fill=(255, 255, 255, 230))
    draw.ellipse([(62, 64), (70, 72)], fill=(255, 255, 255, 255)) # reflection

    # Sparkle / Magic Star at top right
    star_center = (96, 34)
    sx, sy = star_center
    draw.line([(sx, sy - 10), (sx, sy + 10)], fill=(255, 220, 50, 255), width=3)
    draw.line([(sx - 10, sy), (sx + 10, sy)], fill=(255, 220, 50, 255), width=3)
    draw.line([(sx - 6, sy - 6), (sx + 6, sy + 6)], fill=(255, 220, 50, 255), width=2)
    draw.line([(sx - 6, sy + 6), (sx + 6, sy - 6)], fill=(255, 220, 50, 255), width=2)

    img.save(output_path, 'PNG')
    print(f"Icon generated successfully at {output_path}")

if __name__ == '__main__':
    generate_module_icon('static/description/icon.png')
