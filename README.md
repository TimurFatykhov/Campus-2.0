(Ссылка)[https://drive.google.com/open?id=1FwE2qus9yvw507Ym8Y6JUiP3YPce84FX] на размеченные мозги

# Campus-2.0

```
import glob 
import PIL
import numpy as np

def __center_crop__(pil_img):
    width, height = pil_img.size
    diff = height - width

    if diff < 0:
        # width > height
        print('w > h')
        diff = - diff
        half = diff // 2
        pil_img = pil_img.crop((half, 0, width - diff + half, height))

    elif diff > 0:
        # height > width
        half = diff // 2
        pil_img = pil_img.crop((0, half, width, height - diff + half))
        
    return pil_img


to_imgs = sorted(glob.glob('./Жесты(0-9)/*/*/*'))

size = (256, 256)

size = (256, 256)

for to_img in to_imgs:
    img = PIL.Image.open(to_img)
    cropped = __center_crop__(img)
    img = img.resize(size=size, resample=PIL.Image.BICUBIC)
    
    img.save(to_img)
```
