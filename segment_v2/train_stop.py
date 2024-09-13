

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
        mean=total / self.count
        if  mean > self.best:
            self.best = mean

        # 分数没有超过之前，已经 count 次，就停止
        if self.best > score:
            self.trigger_count += 1
            if self.trigger_count > self.count+1:
                return True

        return False


if __name__ == '__main__':
    from tqdm import tqdm
    import time
    # 假设我们有一个很长的任务列表
    tasks = range(100)
    pbar= tqdm(tasks)
    # 使用tqdm创建一个进度条
    for i in pbar:
        # 在循环中输出日志信息
        time.sleep(0.1)
        # pbar.set_description(f"任务{i}: 正在处理...")
        pbar.set_postfix_str(f"任务{i}: 正在处理...")

    # test_list = [1, 2, 3, 3, 3,3, 3, 2, 3, 8, 5, 5, 1, 1,1]
    #
    # train_stop=TrainStop(3)
    # for score in test_list:
    #     is_stop = train_stop(score)
    #     print(f"score: {train_stop.score_list}")
    #     if is_stop:
    #         print(f"stop: {train_stop.best}")
