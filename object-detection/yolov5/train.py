import torch
import torchvision
from colorama import Fore
from torch.optim import Adam
from torch.utils.data import DataLoader
from data import CocoDataset
from net import YoloV5
from loss import YoloV5Loss
import utils
from  tqdm import  trange,tqdm
import logging

# 训练停止条件，连续 多少次没有增长
class TrainStop:
    def __init__(self, count=3):
        self.count = count
        self.score_list = [0.0]
        self.best = 0.0
        self.trigger_count = 0

    def __call__(self, score: float) -> bool:
        self.score_list.append(score)
        total = sum(self.score_list[-self.count:])
        # 最佳分数： 最后几次平均分
        mean = total / self.count
        if mean > self.best:
            self.best = mean

        # 分数没有超过之前，已经 count 次，就停止
        if self.best > score:
            self.trigger_count += 1
            if self.trigger_count > self.count + 1:
                return True

        return False


def config_logger(name="train"):
    # 设置日志的基本配置
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # # 再创建一个handler，用于输出到控制台
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    return logger

logger = config_logger()
train_auto_stop = TrainStop(count=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_path = "coco128/images/train2017"
label_path = "coco128/labels/train2017"
train_dataset = CocoDataset(image_path, label_path)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
eval_dataloader = DataLoader(dataset=train_dataset, batch_size=1)

logger.info(f"训练数据: {len(train_dataloader)}  评估练数据: {len(eval_dataloader)} ")

model = YoloV5(class_num=80).to(device)
loss = YoloV5Loss(class_num=80).to(device)
optimizer = Adam(model.parameters(), 0.001)

scaler = torch.amp.GradScaler()

for epoch in range(100):

    # 训练--------------------------------------------------------------------
    train_bar = tqdm(train_dataloader, total=len(train_dataloader), leave=True, postfix=Fore.WHITE)
    train_bar.set_description(f"[{epoch}/100]")

    model.train()
    train_total_loss = torch.zeros(4).to(device)

    for step, (image, label) in enumerate(train_bar):

        image, label = image.to(device), label.to(device)

        predict_layer_list = model(image)
        loss_value, loss_detail = loss(predict_layer_list, label)

        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # cuda 混合精度
        # with torch.amp.autocast(device_type="cuda"):
        #     predict_layer_list = model(image)
        #     loss_value, loss_detail = loss(predict_layer_list, label)
        #
        # scaler.scale(loss_value).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()


        train_total_loss += loss_detail
        # 日志
        box_loss, obj_loss, cls_loss, yolo_loss = train_total_loss.cpu().numpy()
        train_bar.set_postfix(yolo_loss=yolo_loss, box_loss=box_loss,obj_loss=obj_loss, cls_loss=cls_loss,)


    # 验证 --------------------------------------------------------------------
    eval_bar = tqdm(eval_dataloader, total=len(eval_dataloader), leave=True,colour="green", postfix=Fore.GREEN)
    eval_bar.set_description(f"[{epoch}/100]")

    model.eval()
    eval_total_loss = torch.zeros(4).to(device)

    with torch.no_grad():
        for step, (image, label) in enumerate(eval_bar):

            image, label = image.to(device), label.to(device)

            predict_layer_list = model(image)

            loss_value, loss_detail = loss(predict_layer_list, label)

            eval_total_loss += loss_detail
            # 日志
            box_loss, obj_loss, cls_loss, yolo_loss = train_total_loss.cpu().numpy()
            eval_bar.set_postfix(yolo_loss=yolo_loss, box_loss=box_loss, obj_loss=obj_loss, cls_loss=cls_loss)

    # 保存模型--------------------------------------------------------------------
    torch.save(model, f"out/yolov5-{epoch}.pth")
    logger.info(f"第epoch:{epoch} eval:{eval_total_loss.cpu().numpy()} train:{train_total_loss.cpu().numpy().tolist()}  pth: yolov5-{epoch}.pth")


    # # 是否停止--------------------------------------------------------------------
    # is_stop = train_auto_stop(-eval_total_loss[3].item())
    # if is_stop:
    #     logger.info(f"停止训练: epoch:{epoch} 最佳loss {train_auto_stop.best} , score_list:{train_auto_stop.score_list}")
    #     break