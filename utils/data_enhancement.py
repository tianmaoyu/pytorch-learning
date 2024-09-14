import random

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms, functional
from torchvision.transforms.v2 import ToPILImage


# from torchvision.transforms.functional import rotate
# from torchvision.transforms.v2.functional import horizontal_flip, vertical_flip

def image_plt_test():
    img = Image.open("./src/18ea61b2-bb1489b4-bDJI_20230908134506_0017_W.jpeg")
    img_90 = img.rotate(90, expand=True)
    img_180 = img.rotate(180)
    img_270 = img.rotate(270, expand=True)
    horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    vertical_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    # horizontal_flip()
    # vertical_flip()
    # rotate()
    plt.figure(figsize=(30, 5))
    plt.subplot(1, 6, 1)
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(1, 6, 2)
    plt.axis("off")
    plt.imshow(img_90)

    plt.subplot(1, 6, 3)
    plt.axis("off")
    plt.imshow(img_180)

    plt.subplot(1, 6, 4)
    plt.axis("off")
    plt.imshow(img_270)

    plt.subplot(1, 6, 5)
    plt.axis("off")
    plt.imshow(horizontal_flip)

    plt.subplot(1, 6, 6)
    plt.axis("off")
    plt.imshow(vertical_flip)
    plt.show()


def image_torch_test():
    image = Image.open("./src/18ea61b2-bb1489b4-bDJI_20230908134506_0017_W.jpeg")
    to_pil_image = ToPILImage()
    to_tensor = transforms.ToTensor()

    image_90 = functional.rotate(image, 90, expand=True)
    image_180 = functional.rotate(image, 180, expand=True)
    image_270 = functional.rotate(image, 270, expand=True)

    h_flip_image = functional.hflip(image)
    v_flip_image = functional.vflip(image)
    # h_flip_image = functional.hflip(to_tensor(image))
    # v_flip_image = functional.vflip(to_tensor(image))

    plt.figure(figsize=(30, 4))
    plt.subplot(1, 6, 1)
    plt.axis("off")
    plt.imshow(to_pil_image(image))

    plt.subplot(1, 6, 2)
    plt.axis("off")
    plt.imshow(to_pil_image(h_flip_image))

    plt.subplot(1, 6, 3)
    plt.axis("off")
    plt.imshow(to_pil_image(v_flip_image))

    plt.subplot(1, 6, 4)
    plt.axis("off")
    plt.imshow(to_pil_image(image_90))

    plt.subplot(1, 6, 5)
    plt.axis("off")
    plt.imshow(to_pil_image(image_180))

    plt.subplot(1, 6, 6)
    plt.axis("off")
    plt.imshow(to_pil_image(image_270))
    plt.show()

class ImageEnhancementTransform:
    # 图像数据进行增强，
    def __init__(self, transform_type_list=["original", "h_flip", "v_flip", "rotate_90", "rotate_180", "rotate_270"]):
        self.transform_type_list = transform_type_list

    def __call__(self, image, mask):

        type = random.choice(self.transform_type_list)
        print(f"转换类型:{type}")

        if type == "original":
            return functional.to_tensor(image) ,functional.to_tensor(mask)
        if type == "h_flip":
            return functional.hflip(image), functional.hflip(mask)
        if type == "v_flip":
            return functional.vflip(image), functional.vflip(mask)
        if type == "rotate_90":
            return functional.rotate(image, 90, expand=True), functional.rotate(mask, 90, expand=True)
        if type == "rotate_180":
            return functional.rotate(image, 180), functional.rotate(mask, 180)
        if type == "rotate_270":
            return functional.rotate(image, 270, expand=True), functional.rotate(mask, 270, expand=True)

image = Image.open(r"E:\语义分割\water_v1\water_v1\JPEGImages\creek0\34.png")
mask = Image.open("./src/labeled_image.png")

enhancement= ImageEnhancementTransform()
to_pil_image = ToPILImage()

image_1,mask_1=enhancement(image,mask)
image_2,mask_2=enhancement(image,mask)
image_3,mask_3=enhancement(image,mask)

plt.figure(figsize=(6,6))

plt.subplot(4,2,1)
plt.imshow(to_pil_image(image_1))

plt.subplot(4,2,2)
plt.imshow(to_pil_image(mask_1))

plt.subplot(4,2,3)
plt.imshow(to_pil_image(image_2))

plt.subplot(4,2,4)
plt.imshow(to_pil_image(mask_2))

plt.subplot(4,2,5)
plt.imshow(to_pil_image(image_3))

plt.subplot(4,2,6)
plt.imshow(to_pil_image(mask_3))

plt.show()