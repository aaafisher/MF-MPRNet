

import torch
import pandas as pd
import joblib
from model.model import MultiHeadModel2
from train_endface import MultiHeadModel
from model.model_kan_hope_transformer import TransformerHOPEKAN
def predict(sample, model_class, model_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\machining_multi_head_weights.pth", aux_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\machining_multi_head_aux.pkl"):
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
    df=pd.read_csv(r'F:\DL-Study\MachiningParameterRecommend\eval_pth\df.csv')
    input_cols_cat = ["加工特征类型", "零件材料"]
    input_cols_num = ["面宽度", "面长度"]
    out_cat_classes = []  # 分类头的类别数量

    for col in output_cols_cat:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders_out[col] = le
        out_cat_classes.append(df[col].nunique())

    model = TransformerHOPEKAN(
        df,
        input_cols_cat,
        num_cat_inputs=len(input_cols_cat),
        num_num_inputs=len(input_cols_num),
        cat_classes=out_cat_classes,
        num_outputs=len(output_cols_num),
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
    pred_cat_list = []
    for out, col in zip(pred_cat, label_encoders_out.keys()):
        cls_idx = torch.argmax(out, dim=1).item()
        le = label_encoders_out[col]
        pred_cat_list.append(le.inverse_transform([cls_idx])[0])

    # -------------------------------
    # 7. 数值输出反归一化
    # -------------------------------
    pred_num_real = scaler_out.inverse_transform(pred_num.numpy())[0]
    pred_num_dict = dict(zip(scaler_out.feature_names_in_, pred_num_real))

    return pred_cat_list, pred_num_dict



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

# ============================
# 示例推理
# ============================

if __name__=="__main__":
    sample = {
        "加工特征类型": "EndFace",
        "零件材料": "不锈钢",
        "面宽度": 40,
        "面长度": 80
    }

    pred_cat, pred_num = predict(sample, model_class=MultiHeadModel2)
    print("分类输出：", pred_cat)
    print("数值输出：", pred_num)