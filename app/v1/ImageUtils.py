def getFirstPixelTopLeft(img):
    for i in range(img.height):
        for j in range(img.width):
            if img.getpixel((j, i)) != 0:
                return (i, j)

def getFirstPixelLeftTop(img):
    for j in range(img.width):
        for i in range(img.height):
            if img.getpixel((j, i)) != 0:
                return (i, j)

def getFirstRow(img):
    for i in range(img.height):
        for j in range(img.width):
            if img.getpixel((j, i)) != 0:
                return i

def getLastRow(img):
    for i in range(img.height - 1, -1, -1):
        for j in range(img.width):
            if img.getpixel((j, i)) != 0:
                return i

def getFirstCol(img):
    for j in range(img.width):
        for i in range(img.height):
            if img.getpixel((j, i)) != 0:
                return j

def getLastCol(img):
    for j in range(img.width - 1, -1, -1):
        for i in range(img.height):
            if img.getpixel((j, i)) != 0:
                return j