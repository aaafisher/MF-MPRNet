import joblib
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
from sklearn.model_selection import train_test_split

csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path=MachiningParameters('profile_planaring')

#列名
input_cols_cat,input_cols_num,output_cols_cat,output_cols_num=MachiningFeaturelabel(machiningfeaturevalue)
#读取完整的数据
data = pd.read_csv(csv_path,encoding='gbk',dtype=str)
#print('data',data)
##将输入字符转换为数字
label_encoders_in = {}  #用于装输入的类别如PartMaterial 、MachiningFeatureType
for col in input_cols_cat:
    le = LabelEncoder()
    # print('df-clo',df[col])
    data[col] = le.fit_transform(data[col].astype(str)) #将字符’铝合金‘这样的转化为数值1、2、3。
    # print('df-clo', df[col])
    label_encoders_in[col] = le #
#将输出字符转换为数字
for col in output_cols_cat:
    le = LabelEncoder()
    # print('df-clo',df[col])
    data[col] = le.fit_transform(data[col].astype(str)) #将字符’铝合金‘这样的转化为数值1、2、3。
    # print('df-clo', df[col])
    label_encoders_in[col] = le #
# data.to_csv('data_transform.csv', index=False, encoding='utf-8-sig')
out_cat_classes = []  # 分类头的类别数量
label_encoders_out = {}
#输出类别标准化
for col in output_cols_cat:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders_out[col] = le
    out_cat_classes.append(data[col].nunique())
train_df, temp_df = train_test_split(
    data,
    test_size=0.45,
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.1,
    random_state=42
)
print('val_df',len(val_df))
print('train_df',len(train_df))
##
input_scaler = StandardScaler()
output_scaler = StandardScaler()

train_df[input_cols_num] = input_scaler.fit_transform(train_df[input_cols_num])
val_df[input_cols_num] = input_scaler.transform(val_df[input_cols_num])
test_df[input_cols_num] = input_scaler.transform(test_df[input_cols_num])

train_df[output_cols_num] = output_scaler.fit_transform(train_df[output_cols_num])
val_df[output_cols_num] = output_scaler.transform(val_df[output_cols_num])
test_df[output_cols_num] = output_scaler.transform(test_df[output_cols_num])
# print('train_set', train_df)
# print('val_df', val_df)
# print('test_df', test_df)
# train_df.to_csv('train_df.csv', index=False, encoding='utf-8-sig')
# val_df.to_csv('val_df.csv', index=False, encoding='utf-8-sig')
# test_df.to_csv('test_df.csv', index=False, encoding='utf-8-sig')


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

train_set = MyDataset(train_df)
val_set = MyDataset(val_df)
test_set = MyDataset(test_df)

BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=TransformerHOPEKAN(
    data,
    input_cols_cat,
    num_cat_inputs=len(input_cols_cat),
    num_num_inputs=len(input_cols_num),
    cat_classes=out_cat_classes,
    num_outputs=len(output_cols_num),
    machiningfeaturevalue=machiningfeaturevalue,
    emb_dim=8,
    d_model=64
    ).to(device)

# criterion_ce = nn.CrossEntropyLoss()
# criterion_mse = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
    optimizer = torch.optim.Adam(list(model.parameters()) + list(unc_loss.parameters()), lr=3*1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_loss_list=[]
    val_loss_list = []
    train_cat_loss_list=[]
    train_num_loss_list=[]
    val_cat_loss_list=[]
    val_num_loss_list=[]
    start = time.time()
    for epoch in range(200):
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
            # print('out_cat', out_cat)
            # print('out_num', out_num)
            loss_cat = 0

            for i, o in enumerate(out_cat):
                loss_cat += criterion_cat(o, y_cat[:, i])
            loss_cat /= len(out_cat)
            loss_num = criterion_num(out_num, y_num)
            #oss = loss_cat * 0.3 + loss_num * 0.7
            #loss = 0.5 * unc_loss(loss_cat, loss_num)
            loss = 0.5 * unc_loss(loss_cat, loss_num)

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
                loss_cat /= len(out_cat)
                loss_num = criterion_num(out_num, y_num)
                #loss=0.15 * loss_cat + 0.15 * loss_num
               # loss = 0.5 * loss_cat + 0.5 * loss_num
                loss=0.5*unc_loss(loss_cat, loss_num)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(
                f"Epoch {epoch + 1}, train_loss ={train_loss:.4f},train_num_loss ={train_num_loss:.4f},train_cat_loss ={train_cat_loss:.4f},val_loss ={val_loss:.4f}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_cat_loss_list.append(train_cat_loss)
        train_num_loss_list.append(train_num_loss)
        # val_cat_loss_list.append(val_loss)
        # val_num_loss_list.append(val_loss)
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
    loss_dict = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_cat_loss": train_cat_loss_list,
        "train_num_loss": train_num_loss_list
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣\loss"
    # ✅ 保存文件
    save_path = os.path.join(save_dir, f"loss_acc_{timestamp}.json")

    with open(save_path, "w") as f:
        json.dump(loss_dict, f)

    print(f"已保存到: {save_path}")
    return  loss_dict

def train_cat_acc():
    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
    unc_loss = UncertaintyWeightedLoss(num_tasks=2, device=device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(unc_loss.parameters()), lr=3 * 1e-4, weight_decay=6*1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_cat_list=[]
    val_cat_list=[]
    start = time.time()
    for epoch in range(200):
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
            loss_cat /= len(out_cat)
            loss_num = criterion_num(out_num, y_num)
            loss = 0.5 * unc_loss(loss_cat, loss_num)
            #loss = loss_cat + loss_num
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
        val_loss=0
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
                loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
                loss_cat /= len(out_cat)
                loss_num = criterion_num(out_num, y_num)
                # loss=0.15 * loss_cat + 0.15 * loss_num
                # loss = 0.5 * loss_cat + 0.5 * loss_num
                loss = 0.5 * unc_loss(loss_cat, loss_num)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        val_cat_acc = val_correct / val_total
        print(
            f"Epoch {epoch + 1}, train_acc ={train_acc:.4f},val_cat_acc ={val_cat_acc:.4f}")
        train_cat_list.append(train_acc)
        val_cat_list.append(val_cat_acc)
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
    cat_acc_dict = {
        "train_cat_acc": train_cat_list,
        "val_cat_acc": val_cat_list
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir1 = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣\cat_acc"
    # ✅ 保存文件
    save_path1 = os.path.join(save_dir1, f"loss_acc_{timestamp}.json")

    with open(save_path1, "w") as f:
        json.dump(cat_acc_dict, f)
    print(f"已保存到: {save_path1}")
    return cat_acc_dict
###
def train_num_r2():

    criterion_num = nn.MSELoss()
    criterion_cat = nn.CrossEntropyLoss()
    unc_loss = UncertaintyWeightedLoss(num_tasks=2, device=device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(unc_loss.parameters()), lr=5 * 1e-4,
                                 weight_decay=6 * 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_num_R2=[]
    val_num_R2=[]
    start = time.time()
    for epoch in range(200):
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
            loss_cat = 0
            for i, o in enumerate(out_cat):
                loss_cat += criterion_cat(o, y_cat[:, i])
            loss_cat /= len(out_cat)
            loss = unc_loss(loss_cat, loss_num)
            loss.backward()
            # loss_num.backward()
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
        model.eval()
        val_loss=0
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
                loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
                loss_cat /= len(out_cat)
                loss_num = criterion_num(out_num, y_num)

                loss = 0.5 * unc_loss(loss_cat, loss_num)
                val_loss += loss.item()
                # val_loss += loss_num.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        val_preds = torch.cat(val_preds, dim=0).numpy()
        val_targets = torch.cat(val_targets, dim=0).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        val_num_R2.append(val_r2)
        print(f"Epoch {epoch + 1}: Train R² = {train_r2:.4f}, Val R² = {val_r2:.4f}")
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")
    R2_dict = {
        "train_R2": train_num_R2,
        "val_R2": val_num_R2
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir1 = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣\R2"
    # ✅ 保存文件
    save_path1 = os.path.join(save_dir1, f"loss_acc_{timestamp}.json")

    with open(save_path1, "w") as f:
        json.dump(R2_dict, f)
    print(f"已保存到: {save_path1}")
    return R2_dict


def predict(sample, model_path=model_save_pth, aux_path=aux_save_path):

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
        data,
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

    with torch.no_grad():
        pred_cat, pred_num = model(x_cat, x_num)

    # -------------------------------
    pred_cat_list = []
    for out, col in zip(pred_cat, output_cols_cat):
        cls_idx = torch.argmax(out, dim=1).item()
        le = label_encoders_out[col]
        pred_cat_list.append(le.inverse_transform([cls_idx])[0])

    # 6. 数值输出反归一化
    pred_num_real = scaler_out.inverse_transform(pred_num.numpy())[0]
    pred_num_dict = dict(zip(output_cols_num, pred_num_real))

    return pred_cat_list, pred_num_dict

if __name__=="__main__":
    # print('input_cols_cat',input_cols_cat)
    # #print('data', data)
    #print('train_set', train_df)
    #loss_dict = cal_loss()
    # sample = {
    #     "MachiningFeatureType": "Profile",
    #     "PartMaterial": "钢",
    #     "Size_Length": 32,
    #     "Size_Width": 20,
    #     "Size_Height": 20,
    #     "Size_MinConcaveR": 2
    # }
    # pred_cat, pred_num = predict(sample)
    # print("分类输出：", pred_cat)
    # print("数值输出：", pred_num)
    ###
    #cat_acc_dict=train_cat_acc()
    R2_dict=train_num_r2()


