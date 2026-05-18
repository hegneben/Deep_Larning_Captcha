from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# Basis-Einstellungen
width, height = 400, 150
image = Image.new("RGB", (width, height), color=(180, 190, 200)) # Graublauer Hintergrund
draw = ImageDraw.Draw(image)

# Text einfügen (Nutze einen Standard-Font deines Systems)
try:
    font = ImageFont.truetype("arial.ttf", 80)
except IOError:
    font = ImageFont.load_default()

# Text 'ende' zentriert zeichnen (in Dunkelgrün)
draw.text((100, 25), "finish", fill=(0, 100, 0), font=font)

# Störlinien hinzufügen (wie im Beispiel)
# Grüne Wellenlinie unten
draw.line([(0, 110), (100, 100), (200, 120), (300, 100), (400, 110)], fill=(0, 80, 0), width=5)
# Rote Wellenlinie oben
draw.line([(0, 60), (120, 50), (220, 40), (320, 30), (400, 25)], fill=(150, 50, 50), width=4)
# Kleine lila Störlinie links
draw.line([(40, 30), (60, 80)], fill=(70, 50, 120), width=5)

# Optionale leichte Verzerrung für den Captcha-Effekt
image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Bild speichern
image.save("captcha_finish.png")
print("Captcha wurde als 'captcha_ende.png' gespeichert.")
