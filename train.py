import os
import torch
import tqdm
import Unet_base
from torch.utils import data
from torch import nn
from torch import optim
import data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # 网络实例化
    batch_size = 16  # 指定批处理大小

    num_epochs = 100
    device = 'cuda'
    save_loss_min = 100
    weights_path = './weights_base_v1'

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model = Unet_base.UNet(1, [32, 48, 64, 96, 128], 1, net_mode='2d').to(device)

    data_train, data_val, _ = data_loader.get_data_path()

    train_loader = data.DataLoader(data_loader.Dataset(data_train), batch_size=batch_size,
                                   shuffle=True, num_workers=4)

    val_loader = data.DataLoader(data_loader.Dataset(data_val), batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        dt_size = len(train_loader.dataset)
        dt_size_val = len(val_loader.dataset)
        epoch_loss = 0
        pbar = tqdm.tqdm(
            total=dt_size // batch_size,
            desc=f'Epoch {epoch + 1} / {num_epochs}',
            postfix=dict,
            miniters=.3
        )
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{
                'train_loss': epoch_loss / (i + 1),
            })
            pbar.update(1)
        pbar.close()
        pbar = tqdm.tqdm(
            total=dt_size_val // batch_size,
            desc=f'Val_Epoch {epoch + 1} / {num_epochs}',
            postfix=dict,
            miniters=.3
        )
        epoch_loss_val = 0
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            epoch_loss_val += loss.item()
            pbar.set_postfix(**{
                'val_loss': epoch_loss_val / (i + 1),
            })
            pbar.update(1)
        pbar.close()
        if save_loss_min > epoch_loss_val / i:
            save_loss_min = epoch_loss_val / i
            torch.save(model.state_dict(), weights_path + '/weights.pth')
    print("训练完成！")
