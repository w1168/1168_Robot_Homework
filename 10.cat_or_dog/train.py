#此代码由ai生成，仅用于学习，并非本人创作

import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

def load_data(data_dir, data_transforms):
    """
    加载数据集并创建数据加载器。
    
    参数:
    - data_dir: 数据集目录路径。
    - data_transforms: 数据预处理转换字典，包含训练和测试数据的预处理步骤。
    
    返回:
    - image_datasets: 图像数据集字典，包含训练和测试数据集。
    - data_loader: 数据加载器字典，包含训练和测试数据加载器。
    """
    try:
        image_datasets = {
            x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                    transform=data_transforms[x])
            for x in ["train", "test"]
        }
        data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=20, shuffle=True) for x in ["train", "test"]}
        return image_datasets, data_loader
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

def train_model(model, data_loader, loss_f, optimizer, num_epochs, device, save_path):
    """
    训练模型并在训练和测试数据上评估性能。
    
    参数:
    - model: 要训练的模型。
    - data_loader: 数据加载器字典，包含训练和测试数据加载器。
    - loss_f: 损失函数。
    - optimizer: 优化器。
    - num_epochs: 训练的轮数。
    - device: 运行模型的设备，可以是'cuda'或'cpu'。
    - save_path: 保存模型的路径。
    """
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        for phase in ["train", "test"]:
            try:
                if phase == "train":
                    print("training")
                    model.train(True)
                else:
                    print("testing")
                    model.train(False)
                
                running_loss = 0.0
                running_corrects = 0
                
                for batch, data in enumerate(data_loader[phase], 1):
                    X, y = data
                    X, y = X.to(device), y.to(device)
                    
                    y_pred = model(X)
                    _, pred = torch.max(y_pred.data, 1)
                    optimizer.zero_grad()
                    loss = loss_f(y_pred, y)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item()
                    running_corrects += torch.sum(pred == y.data)
                    
                    if batch % 5000 == 0 and phase == "train":
                        print("Batch {}, Train Loss: {:.4f}, Train ACC: {:.4f}".format(batch, running_loss / batch, 100 * running_corrects / (20 * batch)))
                
                epoch_loss = running_loss * 20 / len(image_datasets[phase])
                epoch_acc = 100 * running_corrects / len(image_datasets[phase])
                print("{} Loss: {:.4f} Acc: {:.4f}%".format(phase, epoch_loss, epoch_acc))
                
                # 在每个epoch结束时保存模型
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch}.pth'))
                    print(f"Model saved at epoch {epoch} with accuracy {best_acc:.4f}%")
            
            except RuntimeError as e:
                print(f"Runtime error during {phase}: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred during {phase}: {e}")
                continue

# 主程序
if __name__ == "__main__":
    # 定义数据目录和数据预处理转换
    data_dir = "./data"
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
    }

    # 加载数据集和创建数据加载器
    image_datasets, data_loader = load_data(data_dir, data_transforms)

    # 加载预训练的ResNet-50模型并修改最后一层(false)
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(2048, 2)

    # 检查是否有可用的GPU并将其用于模型
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.00001)

    # 定义训练轮数和模型保存路径
    num_epochs = 4
    save_path = "./models"

    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 训练模型
    train_model(model, data_loader, loss_f, optimizer, num_epochs, device, save_path)
    """
    训练模型并在训练和测试数据上评估性能。
    
    参数:
    - model: 要训练的模型。
    - data_loader: 数据加载器字典，包含训练和测试数据加载器。
    - loss_f: 损失函数。
    - optimizer: 优化器。
    - num_epochs: 训练的轮数。
    - device: 运行模型的设备，可以是'cuda'或'cpu'。
    - save_path: 保存模型的路径。
    """

    print("over")