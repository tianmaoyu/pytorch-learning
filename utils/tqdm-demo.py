import sys

from  tqdm import  trange,tqdm
from colorama import Fore, Back, Style
import  time
import  logging
logging.basicConfig(filename="app.log",
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tqdm-demo")

list=[i for i in range(10)]

for epoch  in range(5):
    tqdm_bar= tqdm(list,total=len(list),leave=True, postfix=Fore.GREEN)
    for index, item in enumerate(tqdm_bar) :
        time.sleep(0.5)
        logger.info(f"日志：{index}")
        tqdm_bar.set_description(f"train")
        tqdm_bar.set_postfix(epoch=epoch,loss=0.11,var_miou=0.77,train_miou=0.88)




