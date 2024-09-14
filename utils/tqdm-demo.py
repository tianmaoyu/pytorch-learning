import sys

from  tqdm import  trange,tqdm
import  time
import  logging
logging.basicConfig(filename="app.log",
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tqdm-demo")

list=[i for i in range(10)]

for epoch  in range(5):
    tqdm_bar= tqdm(list,total=len(list),leave=True)
    for item in tqdm_bar:
        time.sleep(0.5)
        logger.info(f"日志：{item}")
        tqdm_bar.set_description(f"epoch:{epoch}/{20}")
        tqdm_bar.set_postfix(loss=0.11,var_miou=0.77,train_miou=0.88)




