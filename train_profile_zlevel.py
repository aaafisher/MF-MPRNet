import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from model.model import MultiHeadModel2
from model.model_kan_hope_transformer import TransformerHOPEKAN
# ==========================================================
# 1. 输入 / 输出列（你已经给出，直接使用）
# ==========================================================
input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
input_cols_num = ["Size_Length", "Size_Width","Size_Height","Size_MinConcaveR"]
##分类

output_cols_cat=["ToolMaterial","ToolType","MachiningProcess","MachiningMethod","Axis",
                 "CutDirection","Cutting_Sequence","LayerToLayer",  "EngageType","ToolPosition"]
#数值

output_cols_num=["ToolDiameter","LowerRadius","ToolLength","FluteLength",
                 "CuttingDepth","PartStock","FinalFloorStock","Extension","Tolerance","SlopeAngle","EngageLength","SpindleSpeed",
                 "FeedRates"]

csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-Zlevel.csv'  # 修改为你的数据路径

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
BATCH_SIZE = 16
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# ==========================================================
# 4. 多头模型：11 分类 + 15 数值回归
# ==========================================================
class MultiHeadModel(nn.Module):
    def __init__(self, num_cat_inputs, num_num_inputs, cat_classes, num_outputs):
        super().__init__()

        # 类别 embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(df[col].nunique(), 8)
            for col in input_cols_cat
        ])

        self.fc_in = nn.Linear(num_cat_inputs * 8 + num_num_inputs, 128)
        self.fc_mid = nn.Linear(128, 64)

        # 11 个分类头
        self.class_heads = nn.ModuleList([
            nn.Linear(64, c) for c in cat_classes
        ])

        # 数值回归头：一个 64→15 的线性层
        self.reg_head = nn.Linear(64, num_outputs)

    def forward(self, x_cat, x_num):
        embeds = []
        for i, emb in enumerate(self.embeddings):
            embeds.append(emb(x_cat[:, i]))
        x_cat_emb = torch.cat(embeds, dim=1)

        x = torch.cat([x_cat_emb, x_num], dim=1)

        x = torch.relu(self.fc_in(x))
        x = torch.relu(self.fc_mid(x))

        # 多个分类输出
        out_cat = [head(x) for head in self.class_heads]

        # 数值输出
        out_num = self.reg_head(x)

        return out_cat, out_num



model=TransformerHOPEKAN(
    df,
    input_cols_cat,
    num_cat_inputs=len(input_cols_cat),
    num_num_inputs=len(input_cols_num),
    cat_classes=out_cat_classes,
    num_outputs=len(output_cols_num),
    machiningfeaturevalue='profile_zlevel',
    emb_dim=8,
    d_model=64
    )

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 5. Train
# ==========================================================
def train_one_epoch():
    total_loss = 0
    model.train()

    for x_cat, x_num, y_cat, y_num in loader:
        optimizer.zero_grad()

        pred_cat, pred_num = model(x_cat, x_num)

        # 分类损失（11 个头）
        loss_cat = 0
        for i in range(len(pred_cat)):
            loss_cat += criterion_ce(pred_cat[i], y_cat[:, i])

        # 回归损失
        loss_num = criterion_mse(pred_num, y_num)

        loss = loss_cat + loss_num
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# 训练
# for epoch in range(10):
#     loss = train_one_epoch()
#     print(f"Epoch {epoch+1}, Loss={loss:.4f}")

# 6. 训练循环 + 验证
# -------------------------------
def train():

    EPOCHS=5
    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x_cat, x_num, y_cat, y_num in train_loader:
            optimizer.zero_grad()
            out_cat, out_num = model(x_cat, x_num)
            loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))

            print("y_num和：",np.isnan(y_num).sum())
            loss_num = criterion_num(out_num, y_num)
            loss = loss_cat + loss_num

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_cat.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y_cat, y_num in val_loader:
                out_cat, out_num = model(x_cat, x_num)
                loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
                loss_num = criterion_num(out_num, y_num)
                loss = loss_cat + loss_num
                val_loss += loss.item() * x_cat.size(0)
            val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_cat, x_num, y_cat, y_num in test_loader:
            out_cat, out_num = model(x_cat, x_num)
            loss_cat = sum(criterion_cat(o, y_cat[:, i]) for i, o in enumerate(out_cat))
            loss_num = criterion_num(out_num, y_num)
            loss = loss_cat + loss_num
            test_loss += loss.item() * x_cat.size(0)
    test_loss /= len(test_loader.dataset)
    print("Test Loss:", test_loss)
# -----
# ==========================================================
# 6. Save Model
# ==========================================================
torch.save(model.state_dict(), r"F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-zlevel_weights.pth")
import joblib
joblib.dump({
    "label_encoders_in": label_encoders_in,
    "label_encoders_out": label_encoders_out,
    "scaler_in": scaler_in,
    "scaler_out": scaler_out
}, r"F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-zlevel_aux.pkl")

print("模型保存完成：Profile-zlevel_weights.pth")

def predict(sample, model_class, model_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-zlevel_weights.pth", aux_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\Profile-zlevel_aux.pkl"):
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
        machiningfeaturevalue='profile_zlevel',
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
    train()

    # sample = {
    #     "MachiningFeatureType": "Profile",
    #     "PartMaterial": "钢",
    #     "Size_Length": 32,
    #     "Size_Width": 20,
    #     "Size_Height": 20,
    #     "Size_MinConcaveR": 2
    # }
    # pred_cat, pred_num = predict(sample, model_class=MultiHeadModel2)
    # print("分类输出：", pred_cat)
    # print("数值输出：", pred_num)
