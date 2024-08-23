from torch.utils.tensorboard import SummaryWriter
from PIL import  Image
import  numpy

writer = SummaryWriter("logs")

image=Image.open("data/train/ants/kurokusa.jpg")

image= numpy.array(image)
# dataformats 图片 类型 h w c channel
writer.add_image("image-test",image,2,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2*x",2*i,i)

writer.close()
