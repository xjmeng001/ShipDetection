import glob

from PIL import Image
import os

from yolo import YOLO

yolo = YOLO()

path = 'E:\\Graduation\\1'
# files = os.listdir(path)
# print(files)
image_path = os.path.join(path, "images")

for file in os.listdir(image_path):
    img_name = os.path.splitext(file)[0]
    img = os.path.join(image_path, file)
    print(img)
    image = Image.open(img)
    r_image = yolo.detect_image(image)
    r_image.save(path + '\\predict\\' + img_name + '.jpg')
    # print(img_name)

# for img in glob.glob('{}/*jpg'.format(path)):
#     r_image = yolo.detect_image(img)
#     r_image.save('img.jpg')
#     r_image.show()
