import pyexpat, time, xml.etree.ElementTree as ET
print(pyexpat.EXPAT_VERSION)
lol = '<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaaaaaaaa">' + "".join(
    f'<!ENTITY {chr(98+i)} "' + ("&" + chr(97+i) + ";") * 10 + '">' for i in range(8)) + ']><l>&i;</l>'
t = time.perf_counter()
try:
    ET.fromstring(lol); print("parsed")
except ET.ParseError as e:
    print("refused:", e)
print(f"{time.perf_counter()-t:.3f}s")
