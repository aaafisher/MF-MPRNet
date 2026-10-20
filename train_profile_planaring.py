import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from cat_num_label import MachiningFeaturelabel
from model.model_kan_hope_transformer import TransformerHOPEKAN
import os
import json
import time
# ==========================================================
# 1. 输入 / 输出列（你已经给出，直接使用）
# ==========================================================
# input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
# input_cols_num = ["Size_Length", "Size_Width","Size_Height","Size_MinConcaveR"]
# ##分类
#
# output_cols_cat=["ToolMaterial","ToolType","MachiningProcess","MachiningMethod","Axis","CutPattern","Stepover",
#                  "CutDirection",  "EngageType","ToolPosition"]
# #数值
#
# output_cols_num=["ToolDiameter","LowerRadius","ToolLength","FluteLength","DiameterPercentage",
#                  "CuttingDepth","PartStock","FinalFloorStock","Tolerance","EngageLength","SpindleSpeed",
#                  "FeedRates"]

csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-planarMilling.csv'  # 修改为你的数据路径
machiningfeaturevalue='profile_planaring'
input_cols_cat,input_cols_num,output_cols_cat,output_cols_num=MachiningFeaturelabel(machiningfeaturevalue)
# ==========================================================
# 2. 读取数据
# ==========================================================
df = pd.read_csv(csv_path,encoding='gbk',dtype=str)

# -----------------------------
# 处理输入类别特征
# -----------------------------
label_encoders_in = {}
for col in input_cols_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders_in[col] = le

# -----------------------------
# 数值特征标准化
# -----------------------------
scaler_in = StandardScaler()
df[input_cols_num] = scaler_in.fit_transform(df[input_cols_num])

# -----------------------------
# 处理输出分类标签
# -----------------------------
label_encoders_out = {}
out_cat_classes = []  # 分类头的类别数量

for col in output_cols_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders_out[col] = le
    out_cat_classes.append(df[col].nunique())

# 输出数值的 scaler（回归用）
scaler_out = StandardScaler()
df[output_cols_num] = scaler_out.fit_transform(df[output_cols_num])

# ==========================================================
# 3. Dataset
# ==========================================================
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

        return x_cat, x_num, y_cat, y_num
#df.to_csv(r'F:\DL-Study\MachiningParameterRecommend\eval_pth\df.csv', index=False, encoding='utf-8-sig')
#print('df---',type(df))
dataset = MyDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
##
# 划分训练/验证/测试
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
BATCH_SIZE = 8
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# ==========================================================



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
    )
# print('input_cols_cat',len(input_cols_cat))
# print('input_cols_num',len(input_cols_num))
# print('out_cat_classes',len(out_cat_classes))
# print('output_cols_num',len(output_cols_num))

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================

def compute_y_stats(train_loader):
    all_y = []

    for _, _, _, y_num in train_loader:
        all_y.append(y_num)

    all_y = torch.cat(all_y, dim=0)

    y_mean = all_y.mean(dim=0)
    y_std  = all_y.std(dim=0)

    # 防止除0
    y_std[y_std == 0] = 1

    return y_mean, y_std
# 6. 训练循环 + 验证
# -------------------------------

def train(y_mean, y_std):
    save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率"
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在就创建
    train_num_loss_list = []
    train_cat_loss_list = []
    train_cat_acc_list=[]
    train_num_acc_list = []
    val_cat_loss_list = []
    val_num_loss_list = []

    train_acc_list = []
    val_acc_list = []
    EPOCHS=10
    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
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
        train_num_acc_sum = 0
        #
        abs_error1=0
        train_count = 0
        train_abs_error=0


        for x_cat, x_num, y_cat, y_num in train_loader:
            optimizer.zero_grad()
            out_cat, out_num = model(x_cat, x_num)
            loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))

            #print("y_num和：",np.isnan(y_num).sum())
            loss_num = criterion_num(out_num, y_num)
            loss = loss_cat + loss_num

            loss.backward()


            optimizer.step()
            #train_loss += loss.item() * x_cat.size(0)
            #train_loss /= len(train_loader.dataset)
            train_num_loss += loss_num.item() * x_cat.size(0)
            train_cat_loss += loss_cat.item() * x_cat.size(0)
            train_num_loss /=len(train_loader.dataset)
            train_cat_loss /=len(train_loader.dataset)
            for i, o in enumerate(out_cat):
                pred = o.argmax(dim=1)
                train_correct += (pred == y_cat[:, i]).sum().item()
                train_total += y_cat.size(0)
            # ===== ✅ 数值预测准确率（反归一化后） =====
        #print(f"Epoch {epoch+1},  train_cat_loss ={train_cat_loss:.4f},train_num_loss ={train_num_loss:.4f}")
        abs_error1 = torch.abs(out_num - y_num)
        abs_error1=1-abs_error1
        train_abs_error += abs_error1.sum().item()
        train_count += abs_error1.numel()
        train_value_acc = train_abs_error / train_count
        #print('训练数值准确率：',train_value_acc)
            # y_mean_device = y_mean.to(out_num.device)
            # y_std_device = y_std.to(out_num.device)
            #
            # out_num_real = out_num * y_std_device + y_mean_device
            # y_num_real = y_num * y_std_device + y_mean_device
            #
            # rel_error = torch.abs(out_num_real - y_num_real) / (torch.abs(y_num_real) )

        #     num_acc = 1 - rel_error
        #     num_acc = torch.clamp(num_acc, min=0.0)
        #
        #     train_num_acc_sum += num_acc.sum().item()
        #     train_count += num_acc.numel()
        train_acc = train_correct / train_total

        print(f"Epoch {epoch + 1}, train_cat_loss ={train_cat_loss:.4f},train_num_loss ={train_num_loss:.4f},train_cat_acc ={train_acc:.4f}, train_num_acc ={train_value_acc:.4f}")
        train_cat_loss_list.append(train_cat_loss)
        train_num_loss_list.append(train_num_loss)
        train_cat_acc_list.append(train_acc)
        train_num_acc_list.append(train_value_acc)
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
                abs_error = torch.abs(out_num - y_num)
                abs_error=1-abs_error
                val_abs_error += abs_error.sum().item()
                val_count += abs_error.numel()
        val_cat_acc = val_correct / val_total
        val_num_acc = val_abs_error / val_count
        #print('验证阶段数值预测准确率：',val_mae)
        print(f"Epoch {epoch+1}, val_cat_loss={val_cat_loss:.4f}, val_num_loss={val_num_loss:.4f},val_cat_acc={val_cat_acc:.4f},val_num_acc={val_num_acc:.4f}")
        # val_loss_list.append(val_cat_loss)
        # val_acc_list.append(val_cat_acc)
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
        "train_num_acc": train_num_acc_list
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
torch.save(model.state_dict(), r"F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-planarMilling_weights.pth")
import joblib
joblib.dump({
    "label_encoders_in": label_encoders_in,
    "label_encoders_out": label_encoders_out,
    "scaler_in": scaler_in,
    "scaler_out": scaler_out
}, r"F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-planarMilling_aux.pkl")

print("模型保存完成：machining_multi_head.pth")

def predict(sample, model_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-planarMilling_weights.pth", aux_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-planarMilling_aux.pkl"):
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
    # model = model_class(
    #     df,
    #     input_cols_cat,
    #     num_cat_inputs=len(label_encoders_in),
    #     num_num_inputs=scaler_in.scale_.shape[0],
    #     cat_classes=cat_classes,
    #     num_outputs=scaler_out.scale_.shape[0]
    # )
    model = TransformerHOPEKAN(
        df,
        input_cols_cat,
        num_cat_inputs=len(input_cols_cat),
        num_num_inputs=len(input_cols_num),
        cat_classes=out_cat_classes,
        num_outputs=len(output_cols_num),
        machiningfeaturevalue='profile_planaring',
        emb_dim=8,
        d_model=64
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # -------------------------------
    # 3. 处理输入类别
    # -------------------------------
    x_cat_vals = [label_encoders_in[col].transform([sample[col]])[0] for col in label_encoders_in.keys()]
    x_cat = torch.tensor([x_cat_vals], dtype=torch.long)

    # -------------------------------
    # 4. 处理输入数值
    # -------------------------------
    num_vals = [[sample[col] for col in scaler_in.feature_names_in_]]  # 保持训练时列顺序
    x_num_vals = scaler_in.transform(num_vals)[0]
    x_num = torch.tensor([x_num_vals], dtype=torch.float32)

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
    # train(y_mean, y_std)

    sample = {
        "MachiningFeatureType": "Profile",
        "PartMaterial": "钢",
        "Size_Length": 50,
        "Size_Width": 50,
        "Size_Height":12,
        "Size_MinConcaveR": 4
    }
    pred_cat, pred_num = predict(sample)
    print("分类输出：", pred_cat)
    print("数值输出：", pred_num)
