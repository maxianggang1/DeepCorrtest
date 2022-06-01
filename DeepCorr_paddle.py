# image 处理为三维[通道数，宽，高]
# Conv2D 接收四维数据[batch, 通道数， 宽， 高]
import pipeline
from tqdm import tqdm
dataset = pipeline.Dataset(cf='config.yml', cv_i=0,  test_only=False, h5_path='data/tordata300.h5')
train_generator = dataset.train_generator
val_generator = dataset.val_generator
test_generator = dataset.test_generator

desc = 'Training'
pred_only=False
n_steps = 55
print(train_generator.epoch())
import paddle
import paddle.nn as nn
model = paddle.nn.Sequential(
    nn.Conv2D(1, 8, 2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(4800, 1),
)
model.train()
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
import paddle.nn.functional as F
for epoch in range(200):
    with tqdm(train_generator.epoch(), total=n_steps,
              desc=f'{desc} | Epoch {epoch}' if not pred_only else desc) as pbar:
        for step, (x, y_truth) in enumerate(pbar):
            x = x.astype('float32')
            x = paddle.to_tensor(x)
            x = paddle.reshape(x, [256,1, 8, 300])
            y_truth = y_truth.astype('float32')
            y = paddle.to_tensor(y_truth)
            y = paddle.reshape(y, [256, 1])
            predicts = model(x)

            loss = nn.functional.cross_entropy(predicts, y)
            #loss = F.cross_entropy(predicts, y)
            avg_loss = paddle.mean(loss)

            # 每训练了1000批次的数据，打印下当前Loss的情况
            if step % 10 == 0:
                print("epoch_id: {},  loss is: {}".format(epoch, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()






