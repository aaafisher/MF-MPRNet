import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from cat_num_label import MachiningFeaturelabel,MachiningParameters
from model.model_kan_hope_transformer import TransformerHOPEKAN
import os
import json
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
import math
import csv
from 损失及准确率.writing_curve import plot_curve

# 可选7个： profile_planaring、chamfer、endface、drillhole、hole_holemilling、pocket_planaring、profile_zlevel
csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path=MachiningParameters('profile_planaring')

input_cols_cat,input_cols_num,output_cols_cat,output_cols_num=MachiningFeaturelabel(machiningfeaturevalue)

df = pd.read_csv(csv_path,encoding='gbk',dtype=str)
# print('df',df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('df-',df[:3])
class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 输入类别
        x_cat = torch.tensor(row[input_cols_cat].astype(int).values, dtype=torch.long)

        # 输入数值
        x_num = torch.tensor(row[input_cols_num].astype(float).values, dtype=torch.float32)

        # 输出分类
        y_cat = torch.tensor(row[output_cols_cat].astype(int).values, dtype=torch.long)

        # 输出数值（回归）
        y_num = torch.tensor(row[output_cols_num].astype(float).values, dtype=torch.float32)
        # print('x_cat',x_cat)
        # print('x_num', x_num)

        return x_cat, x_num, y_cat, y_num

dataset = MyDataset(df)


loader = DataLoader(dataset, batch_size=32, shuffle=True)
#先划分数据集，再标准化
train_size = int(0.75 * len(dataset))
val_size = int(0.24 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
#标注化
label_encoders_in = {}  #用于装输入的类别如PartMaterial 、MachiningFeatureType
for col in input_cols_cat:
    le = LabelEncoder()
    # print('df-clo',df[col])
    df[col] = le.fit_transform(df[col].astype(str)) #将字符’铝合金‘这样的转化为数值1、2、3。
    # print('df-clo', df[col])
    label_encoders_in[col] = le #
# print('df-clo', df) #之后的df数据中的类别输入全部是数字了
# 数值特征标准化
#scaler_in = StandardScaler()
scaler_in = RobustScaler()
df[input_cols_num] = scaler_in.fit_transform(df[input_cols_num])
# print('df-clo', df) #输入型数值标准，将大的数据统一到0-1区间
# -----------------------------
# 处理输出分类标签
# -----------------------------
label_encoders_out = {}
out_cat_classes = []  # 分类头的类别数量
#输出类别标准化
for col in output_cols_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders_out[col] = le
    out_cat_classes.append(df[col].nunique())

# print('df-clo', df) #所有的数据都变成了数值，包括输出的类别数据
# 输出数值的 scaler（回归用），输出数值标准化
#scaler_out = StandardScaler()
scaler_out = RobustScaler()
df[output_cols_num] = scaler_out.fit_transform(df[output_cols_num])
# print('df-',df[:3])
df.to_csv('output_pd_2d.csv', index=False, encoding='utf-8-sig')

BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model=TransformerHOPEKAN(
    df,
    input_cols_cat,
    num_cat_inputs=len(input_cols_cat),
    num_num_inputs=len(input_cols_num),
    cat_classes=out_cat_classes,
    num_outputs=len(output_cols_num),
    machiningfeaturevalue=machiningfeaturevalue,
    emb_dim=8,
    d_model=64
    ).to(device)

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ==========================================================
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        diff = pred - target
        # 使用公式：log(cosh(x)) = x + log(1 + exp(-2x)) - log(2)  对于 x >= 0
        # 对称处理
        sign = torch.sign(diff)
        x = sign * diff  # 取绝对值，但保留符号用于公式？实际上公式只依赖绝对值
        # 更简单：使用绝对值版本
        abs_diff = torch.abs(diff)
        # 稳定计算: log(cosh(x)) = x + log1p(exp(-2x)) - log(2)
        loss = abs_diff + torch.log1p(torch.exp(-2 * abs_diff)) - math.log(2)
        return loss.mean()
# 6. 训练循环 + 验证
# -------------------------------
class QuantileLoss(nn.Module):
    def __init__(self, tau=0.5):
        """
        tau: 分位数，范围 (0, 1)
        """
        super(QuantileLoss, self).__init__()
        self.tau = tau

    def forward(self, pred, target):
        diff = target - pred
        loss = torch.max(self.tau * diff, (self.tau - 1) * diff)
        return loss.mean()
def train(device):

    os.makedirs(save_dir, exist_ok=True)  # 如果不存在就创建
    train_num_loss_list = []
    train_cat_loss_list = []
    train_cat_acc_list=[]
    train_num_acc_list = []
    val_cat_loss_list = []
    val_num_loss_list = []


    val_cat_acc_list = []
    val_num_acc_list=[]
    EPOCHS=100
    criterion_cat = nn.CrossEntropyLoss()
    #损失函数为MSE
    #criterion_num = nn.MSELoss()
    #损失函数为huber
    #criterion_num = nn.HuberLoss(delta=1.0)
    #损失函数MAE
    criterion_num=nn.L1Loss()
    #Log-Cosh 损失
    #criterion_num=LogCoshLoss()
    # Quantile 损失（分位数损失）,效果最差
    #criterion_num=QuantileLoss(tau=0.9)
    #Poisson 损失
    #criterion_num=nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(EPOCHS):
        model.train()
        #分类任务损失
        #train_loss = 0
        train_cat_loss = 0
        #数值任务损失
        train_num_loss=0

        #分类任务训练准确率
        train_correct = 0
        train_total = 0
        # 数值任务训练准确率


        train_r2_total = 0
        train_batches = 0
        all_preds = []
        all_targets = []
        all_preds2 = []
        all_targets2 = []

        for x_cat, x_num, y_cat, y_num in train_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_cat = y_cat.to(device)
            y_num = y_num.to(device)
            optimizer.zero_grad()
            out_cat, out_num = model(x_cat, x_num)

            loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))

            #print("y_num和：",np.isnan(y_num).sum())
            loss_num = criterion_num(out_num, y_num)

            loss = loss_cat + loss_num

            loss.backward()


            optimizer.step()
            all_preds.append(out_num.detach().cpu())
            all_targets.append(y_num.detach().cpu())
            #train_loss += loss.item() * x_cat.size(0)
            #train_loss /= len(train_loader.dataset)
            train_num_loss += loss_num.item() * x_cat.size(0)
            train_cat_loss += loss_cat.item() * x_cat.size(0)
            train_num_loss /=len(train_loader.dataset)
            train_cat_loss /=len(train_loader.dataset)
            # 计算 MSE
            #train_mse = mean_squared_error( y_num.detach().cpu(),out_num.detach().cpu())
            #计算MSLE
            #msle = mean_squared_log_error(y_num.detach().cpu(),out_num.detach().cpu())

            for i, o in enumerate(out_cat):
                pred = o.argmax(dim=1)
                train_correct += (pred == y_cat[:, i]).sum().item()
                train_total += y_cat.size(0)
            # ===== ✅ 数值预测准确率（反归一化后） =====
        #print(f"Epoch {epoch+1},  train_cat_loss ={train_cat_loss:.4f},train_num_loss ={train_num_loss:.4f}")
        #数值准确率计算，r2
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        train_r2 = r2_score(
            all_targets.numpy(),
            all_preds.numpy()
        )

        #分类准确率计算
        train_acc = train_correct / train_total

        print(f"Epoch {epoch + 1}, train_cat_loss ={train_cat_loss:.4f},train_num_loss ={train_num_loss:.4f},train_cat_acc ={train_acc:.4f}, train_num_acc ={train_r2:.4f}")
        train_cat_loss_list.append(train_cat_loss)
        train_num_loss_list.append(train_num_loss)
        train_cat_acc_list.append(train_acc)
        train_num_acc_list.append(train_r2)
        # 验证
        model.eval()
        #验证阶段，分类任务损失
        val_loss = 0
        #验证，分类准确率
        val_correct = 0
        val_total = 0
        ##验证阶段，数值预测的准确率
        val_abs_error = 0
        val_count = 0
        val_num_loss=0
        val_cat_loss=0
        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in val_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y_cat = y_cat.to(device)
                y_num = y_num.to(device)
                out_cat, out_num = model(x_cat, x_num)
                loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
                loss_num = criterion_num(out_num, y_num)
                #loss = loss_cat + loss_num
                val_num_loss += loss_num.item() * x_cat.size(0)
                val_cat_loss += loss_cat.item() * x_cat.size(0)

                val_num_loss /= len(val_loader.dataset)
                val_cat_loss /= len(val_loader.dataset)
            # ===== ✅ 验证准确率 =====
                for i, o in enumerate(out_cat):
                    pred = o.argmax(dim=1)
                    val_correct += (pred == y_cat[:, i]).sum().item()
                    val_total += y_cat.size(0)
                # ===== ✅ MAE计算 =====
                # abs_error = torch.abs(out_num - y_num)
                # abs_error=1-abs_error
                # val_abs_error += abs_error.sum().item()
                # val_count += abs_error.numel()
                all_preds2.append(out_num.detach().cpu())
                all_targets2.append(y_num.detach().cpu())
        val_cat_acc = val_correct / val_total
        # val_num_acc = val_abs_error / val_count

        all_preds2 = torch.cat(all_preds2, dim=0)
        all_targets2 = torch.cat(all_targets2, dim=0)

        Val_r2 = r2_score(
            all_targets2.numpy(),
            all_preds2.numpy()
        )
        #print('验证阶段数值预测准确率：',val_mae)
        print(f"Epoch {epoch+1}, val_cat_loss={val_cat_loss:.4f}, val_num_loss={val_num_loss:.4f},val_cat_acc={val_cat_acc:.4f},val_num_acc={Val_r2:.4f}")
        val_cat_loss_list.append(val_cat_loss)
        val_num_loss_list.append(val_num_loss)
        val_cat_acc_list.append(val_cat_acc)
        val_num_acc_list.append(Val_r2)

    model.eval()
    #测试阶段，分类任务损失
    test_loss = 0
    #测试阶段，分类任务准确率
    test_correct = 0
    test_total = 0
    #测试阶段，数值预测的准确率
    test_abs_error = 0
    test_count = 0
    test_num_loss=0
    test_cat_loss=0
    with torch.no_grad():
        for x_cat, x_num, y_cat, y_num in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_cat = y_cat.to(device)
            y_num = y_num.to(device)
            out_cat, out_num = model(x_cat, x_num)
            loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
            loss_num = criterion_num(out_num, y_num)
            loss = loss_cat + loss_num
            test_num_loss += loss_num.item() * x_cat.size(0)
            test_cat_loss += loss_cat.item() * x_cat.size(0)
            #test_loss += loss.item() * x_cat.size(0)
            # ===== ✅ 测试准确率 =====
            for i, o in enumerate(out_cat):
                pred = o.argmax(dim=1)
                test_correct += (pred == y_cat[:, i]).sum().item()
                test_total += y_cat.size(0)
            # ===== ✅ MAE计算 =====
            abs_error = torch.abs(out_num - y_num)
            abs_error=1-abs_error
            test_abs_error += abs_error.sum().item()
            test_count += abs_error.numel()

    test_num_loss /= len(test_loader.dataset)
    test_cat_loss /= len(test_loader.dataset)

    test_cat_acc = test_correct / test_total
    test_num_acc = test_abs_error / test_count
    print(
        f"Epoch {epoch + 1}, test_cat_loss={test_cat_loss:.4f}, test_num_loss={test_num_loss:.4f},test_cat_acc={test_cat_acc:.4f},test_num_acc={test_num_acc:.4f}")
    # ✅ 保存内容（包含 loss + acc）
    loss_dict = {
        "train_cat_loss": train_cat_loss_list,
        "train_num_loss": train_num_loss_list,
        "train_cat_acc": train_cat_acc_list,
        "train_num_acc": train_num_acc_list,
        "val_cat_loss":val_cat_loss_list,
        "val_num_loss":val_num_loss_list,
        "val_cat_acc": val_cat_acc_list,
        "val_num_acc":val_num_acc_list
    }



    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # ✅ 保存文件
    save_path = os.path.join(save_dir, f"loss_acc_{timestamp}.json")

    with open(save_path, "w") as f:
        json.dump(loss_dict, f)

    print(f"已保存到: {save_path}")

# -----
# ==========================================================
# 6. Save Model
# ==========================================================
torch.save(model.state_dict(), model_save_pth)
import joblib
joblib.dump({
    "label_encoders_in": label_encoders_in,
    "label_encoders_out": label_encoders_out,
    "scaler_in": scaler_in,
    "scaler_out": scaler_out
}, aux_save_path)

print("模型保存完成：machining_multi_head.pth")

def train_num_r2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_num = nn.MSELoss()
    train_num_R2=[]
    val_num_R2=[]
    start = time.time()
    for epoch in range(10):
        model.train()
        #print("Number of batches in train_loader:", len(train_loader))
        for x_cat, x_num, y_cat, y_num in train_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_cat = y_cat.to(device)
            y_num = y_num.to(device)
            optimizer.zero_grad()
            out_cat, out_num = model(x_cat, x_num)
            loss_num = criterion_num(out_num, y_num)

            loss = loss_num

            loss.backward()

            optimizer.step()
            # ========== 评估训练集 R² ==========
        model.eval()
        train_preds = []
        train_targets = []
        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in train_loader:
                #print("Batch loaded")  #
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y_num = y_num.to(device)
                out_cat, out_num = model(x_cat, x_num)
                train_preds.append(out_num.cpu())
                train_targets.append(y_num.cpu())
                # 检查是否有数据
            if len(train_preds) == 0:
                print(f"Epoch {epoch + 1}: No data in train_loader, skipping R2")
                continue
        train_preds = torch.cat(train_preds, dim=0).numpy()
        train_targets = torch.cat(train_targets, dim=0).numpy()
        train_r2 = r2_score(train_targets, train_preds)
        train_num_R2.append(train_r2)

        # ========== 评估验证集 R² ==========
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in val_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y_num = y_num.to(device)
                out_cat, out_num = model(x_cat, x_num)
                val_preds.append(out_num.cpu())
                val_targets.append(y_num.cpu())
        val_preds = torch.cat(val_preds, dim=0).numpy()
        val_targets = torch.cat(val_targets, dim=0).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        val_num_R2.append(val_r2)
        print(f"Epoch {epoch + 1}: Train R² = {train_r2:.4f}, Val R² = {val_r2:.4f}")
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
        #print('train_r2',train_r2)
    return train_num_R2,val_num_R2

def train_cat_acc():
    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
    train_cat_list=[]
    val_cat_list=[]
    start = time.time()
    for epoch in range(10):
        train_correct=0
        train_total=0
        model.train()
        for x_cat, x_num, y_cat, y_num in train_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_cat = y_cat.to(device)
            y_num = y_num.to(device)
            optimizer.zero_grad()
            out_cat, out_num = model(x_cat, x_num)
            loss_cat = 0
            for i, o in enumerate(out_cat):
                loss_cat += criterion_cat(o, y_cat[:, i])
            loss_num = criterion_num(out_num, y_num)
            loss = loss_cat + loss_num
            loss.backward()
            optimizer.step()

            for i, o in enumerate(out_cat):
                pred = o.argmax(dim=1)
                train_correct += (pred == y_cat[:, i]).sum().item()
                train_total += y_cat.size(0)


        train_acc = train_correct / train_total
        model.eval()
        # 验证阶段，分类任务损失

        val_correct = 0
        val_total = 0

        val_num_loss = 0
        val_cat_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in val_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y_cat = y_cat.to(device)
                y_num = y_num.to(device)
                out_cat, out_num = model(x_cat, x_num)

                # =====  验证准确率 =====
                for i, o in enumerate(out_cat):
                    pred = o.argmax(dim=1)
                    val_correct += (pred == y_cat[:, i]).sum().item()
                    val_total += y_cat.size(0)

        val_cat_acc = val_correct / val_total
        print(
            f"Epoch {epoch + 1}, train_acc ={train_acc:.4f},val_cat_acc ={val_cat_acc:.4f}")
        train_cat_list.append(train_acc)
        val_cat_list.append(val_cat_acc)
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
    return train_cat_list,val_cat_list
class UncertaintyWeightedLoss(nn.Module):
    """
    实现论文 "Multi-Task Learning Using Uncertainty to Weigh Losses"
    损失形式：L = L_cls/(2*sigma1^2) + L_reg/(2*sigma2^2) + log(sigma1*sigma2)
    其中 sigma1, sigma2 是可学习的参数（实际网络中学习 log(sigma^2) 以保证稳定性）
    """
    def __init__(self, num_tasks=2, device='cpu'):
        super().__init__()
        # 初始化 log(sigma^2) 为 0.0，即 sigma ≈ 1
        self.log_var = nn.Parameter(torch.zeros(num_tasks, device=device))

    def forward(self, loss_cls, loss_reg):
        # 两个任务的损失分别加权
        precision_cls = torch.exp(-self.log_var[0])
        precision_reg = torch.exp(-self.log_var[1])
        weighted_loss = precision_cls * loss_cls + precision_reg * loss_reg
        # 加上正则项 log(sigma) = 0.5 * log_var
        reg_term = 0.5 * (self.log_var[0] + self.log_var[1])
        return weighted_loss + reg_term
def cal_loss():

    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
    unc_loss = UncertaintyWeightedLoss(num_tasks=2, device=device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(unc_loss.parameters()), lr=2*1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_loss_list=[]
    val_loss_list = []
    start = time.time()
    for epoch in range(2):
        model.train()

        train_cat_loss = 0
        # 数值任务损失
        train_num_loss = 0
        train_loss=0
        for x_cat, x_num, y_cat, y_num in train_loader:
            #print('x_cat',x_cat)
            #print('x_num', x_num.shape)
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y_cat = y_cat.to(device)
            y_num = y_num.to(device)
            # print('y_cat', y_cat)
            # print('y_num', y_num)
            optimizer.zero_grad()

            out_cat, out_num = model(x_cat, x_num)
            #print('out_cat', out_cat)
            print('out_num', out_num)
            loss_cat = 0

            for i, o in enumerate(out_cat):
                loss_cat += criterion_cat(o, y_cat[:, i])

            loss_num = criterion_num(out_num, y_num)
            #oss = loss_cat * 0.3 + loss_num * 0.7
            #loss = 0.5 * unc_loss(loss_cat, loss_num)
            loss = unc_loss(loss_cat, loss_num)

            loss.backward()

            optimizer.step()
            train_num_loss += loss_num.item()
            train_cat_loss += loss_cat.item()
            #train_loss +=loss.item() * x_cat.size(0)
            train_loss += loss.item()
            train_num_loss /= len(train_loader)
            train_cat_loss /= len(train_loader)
            train_loss = train_loss/len(train_loader)
        # 验证
        model.eval()
        # 验证阶段，分类任务损失
        val_loss = 0

        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in val_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y_cat = y_cat.to(device)
                y_num = y_num.to(device)
                out_cat, out_num = model(x_cat, x_num)
                loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
                loss_num = criterion_num(out_num, y_num)
                #loss=0.15 * loss_cat + 0.15 * loss_num
                loss = 0.5 * loss_cat + 0.5 * loss_num
                val_loss += loss.item()
                val_loss /= len(val_loader)

        print(
                f"Epoch {epoch + 1}, train_loss ={train_loss:.4f},train_num_loss ={train_num_loss:.4f},train_cat_loss ={train_cat_loss:.4f},val_loss ={val_loss:.4f}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
    return train_loss_list,val_loss_list

def predict(sample, model_path=model_save_pth, aux_path=aux_save_path):
    """
    通用预测函数

    参数：
        sample: dict，包含所有输入列，key 为列名，value 为对应值
        model_class: 训练时的 MultiHeadModel 类
        model_path: 模型权重路径
        aux_path: scaler + label encoder 保存路径

    返回：
        pred_cat_list: list，分类列预测结果（反编码）
        pred_num_dict: dict，数值列预测结果（反归一化）
    """

    # -------------------------------
    # 1. 加载辅助对象
    # -------------------------------
    aux = joblib.load(aux_path)
    label_encoders_in = aux["label_encoders_in"]
    label_encoders_out = aux["label_encoders_out"]
    scaler_in = aux["scaler_in"]
    scaler_out = aux["scaler_out"]

    # -------------------------------
    # 2. 初始化模型
    # -------------------------------
    cat_classes = [len(le.classes_) for le in label_encoders_out.values()]

    model = TransformerHOPEKAN(
        df,
        input_cols_cat,
        num_cat_inputs=len(input_cols_cat),
        num_num_inputs=len(input_cols_num),
        cat_classes=out_cat_classes,
        num_outputs=len(output_cols_num),
        machiningfeaturevalue=machiningfeaturevalue,
        emb_dim=8,
        d_model=64
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # -------------------------------
    # 3. 处理输入类别
    # -------------------------------
    x_cat_vals = [label_encoders_in[col].transform([sample[col]])[0] for col in label_encoders_in.keys()]
    x_cat = torch.tensor([x_cat_vals], dtype=torch.long)
    x_cat=x_cat.to(device)
    # -------------------------------
    # 4. 处理输入数值
    # -------------------------------
    num_vals = [[sample[col] for col in scaler_in.feature_names_in_]]  # 保持训练时列顺序
    x_num_vals = scaler_in.transform(num_vals)[0]
    x_num = torch.tensor([x_num_vals], dtype=torch.float32)
    x_num=x_num.to(device)
    # -------------------------------
    # 5. 推理
    # -------------------------------
    with torch.no_grad():
        pred_cat, pred_num = model(x_cat, x_num)

    # -------------------------------
    # 6. 分类输出反编码
    # -------------------------------
    # pred_cat_list = []
    # for out, col in zip(pred_cat, label_encoders_out.keys()):
    #     cls_idx = torch.argmax(out, dim=1).item()
    #     le = label_encoders_out[col]
    #     pred_cat_list.append(le.inverse_transform([cls_idx])[0])
    pred_cat_dict = {}
    for out, col in zip(pred_cat, label_encoders_out.keys()):
        cls_idx = torch.argmax(out, dim=1).item()
        le = label_encoders_out[col]
        pred_cat_dict[col] = le.inverse_transform([cls_idx])[0]
    # -------------------------------
    # 7. 数值输出反归一化
    # -------------------------------
    pred_num_real = scaler_out.inverse_transform(pred_num.numpy())[0]
    pred_num_dict = dict(zip(scaler_out.feature_names_in_, pred_num_real))
    return pred_cat_dict, pred_num_dict
    #return pred_cat_list, pred_num_dict



    # -------------------------------
    with torch.no_grad():
        pred_cat, pred_num = model(x_cat, x_num)

    # -------------------------------
    # 5. 分类头反编码
    # -------------------------------
    pred_cat_list = []
    for out, col in zip(pred_cat, output_cols_cat):
        cls_idx = torch.argmax(out, dim=1).item()
        le = label_encoders_out[col]
        pred_cat_list.append(le.inverse_transform([cls_idx])[0])

    # -------------------------------
    # 6. 数值输出反归一化
    # -------------------------------
    pred_num_real = scaler_out.inverse_transform(pred_num.numpy())[0]
    pred_num_dict = dict(zip(output_cols_num, pred_num_real))

    return pred_cat_list, pred_num_dict


if __name__=="__main__":
    # y_mean, y_std = compute_y_stats(train_loader)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #train(device)
    #回归任务的准确率，即R2指标
    # train_num_R2,val_num_R2=train_num_r2()
    # plot_curve(train_num_R2, val_num_R2, 'numacc')
    #分类任务的准确率ACC
    # train_cat_list,val_cat_list=train_cat_acc()
    # plot_curve(train_cat_list, val_cat_list, 'catacc')
    #损失计算
    train_loss_list,val_loss_list=cal_loss()
    loss_dict = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣\loss"
    # ✅ 保存文件
    save_path = os.path.join(save_dir, f"loss_acc_{timestamp}.json")

    with open(save_path, "w") as f:
        json.dump(loss_dict, f)

    print(f"已保存到: {save_path}")
   # plot_curve(train_loss_list, val_loss_list, 'catloss')
