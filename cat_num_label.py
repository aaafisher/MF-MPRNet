



def MachiningFeaturelabel(Value):
    if Value=="profile_planaring":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
        input_cols_num = ["Size_Length", "Size_Width", "Size_Height", "Size_MinConcaveR"]
        ##分类

        output_cols_cat = ["ToolMaterial", "ToolType", "MachiningProcess", "MachiningMethod", "Axis", "CutPattern",
                           "Stepover",
                           "CutDirection", "EngageType", "ToolPosition"]
        # 数值

        output_cols_num = ["ToolDiameter", "LowerRadius", "ToolLength", "FluteLength", "DiameterPercentage",
                           "CuttingDepth", "PartStock", "FinalFloorStock", "Tolerance", "EngageLength", "SpindleSpeed",
                           "FeedRates"]
        return input_cols_cat,input_cols_num,output_cols_cat,output_cols_num

    elif Value == "chamfer":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
        input_cols_num = ["Size", "Maching_hole_size"]

        output_cols_cat = ["ToolMaterial",
                           "ToolType", "MachiningProcess", "MachiningMethod", "CutPattern",
                           "Stepover", "CutDirection", "EngageType"
                           ]

        output_cols_num = [
            "ToolDiameter", "ToolLength", "FluteLength", "TipAngle", "TaperAngle", "CuttingDepth",
            "DiameterPercentage", "PartStock", "FinalFloorStock", "EngageLength",
            "Tolerance", "SpindleSpeed", "FeedRates"
        ]
        return input_cols_cat, input_cols_num, output_cols_cat, output_cols_num
    elif Value == "endface":
        input_cols_cat = ["加工特征类型", "零件材料"]
        input_cols_num = ["面宽度", "面长度"]

        output_cols_cat = [
            "刀具类型", "Axis", "MachiningProcess", "MachiningMethod", "CutPattern",
            "Stepover", "CutDirection", "ToolOverhang", "EngageType", "EngageLength", "ToolPosition"
        ]

        output_cols_num = [
            "刀具直径", "切削深度", "ToolLength", "FluteLength", "TipAngle", "LowerRadius", "TaperAngle",
            "DiameterPercentage", "StockDistance", "PartStock", "FinalFloorStock", "WallStock",
            "Tolerance", "SpindleSpeed", "FeedRates"
        ]


        return input_cols_cat,input_cols_num,output_cols_cat,output_cols_num
    elif Value == "drillhole":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial", "Through_blind_Hole", "Open_Close_Hole"]
        input_cols_num = ["Hole_Diameter", "Hole_Height"]
        ##分类
        output_cols_cat = ["ToolMaterial", "ToolType", "MachiningProcess", "MachiningMethod", "Axis", "CycleType"]

        # 数值
        output_cols_num = ["ToolDiameter", "ToolLength", "FluteLength", "TipAngle", "TipLength", "CuttingDepth",
                           "SpindleSpeed", "FeedRates"]
        return input_cols_cat, input_cols_num, output_cols_cat, output_cols_num
    elif Value == "hole_holemilling":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial", "Through_blind_Hole"]
        # input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
        input_cols_num = ["Hole_Diameter", "Hole_Height"]
        ##分类
        output_cols_cat = ["ToolMaterial", "ToolType", "MachiningProcess", "MachiningMethod", "CutPattern",
                           "Depth_per_turn", "Axial_pitch", "Radial_pitch", "CutDirection", "EngageType",
                           "Tool_Compensation", "ToolPosition"]

        # 数值
        output_cols_num = ["ToolDiameter", "LowerRadius", "ToolLength", "FluteLength", "Pitch", "ToolPathNumbers",
                           "Max_Distance", "top_bias_distance", "Down_bias_distance", "part_side_clearance",
                           "Tolerance", "EngageLength", "Min_Move", "Min_angle", "SpindleSpeed", "FeedRates"]
        return input_cols_cat, input_cols_num, output_cols_cat, output_cols_num
    elif Value == "pocket_planaring":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial", "Through_blind_pocket", "Open_Close_Pocket"]
        input_cols_num = ["Size_Length", "Size_Width", "Size_Height", "Size_MinConcaveR", "Bottom_R"]
        # 分类
        output_cols_cat = ["ToolMaterial", "ToolType", "MachiningProcess", "MachiningMethod", "Axis",
                           "CutPattern", "Stepover", "CutDirection", "EngageType", "ToolPosition"]

        # 数值
        output_cols_num = ["ToolDiameter", "LowerRadius", "ToolLength", "FluteLength", "DiameterPercentage",
                           "CuttingDepth", "PartStock", "FinalFloorStock", "Tolerance", "Slope_angle",
                           "EngageLength", "SpindleSpeed", "FeedRates"]
        return input_cols_cat, input_cols_num, output_cols_cat, output_cols_num
    elif Value == "profile_zlevel":
        input_cols_cat = ["MachiningFeatureType", "PartMaterial"]
        input_cols_num = ["Size_Length", "Size_Width", "Size_Height", "Size_MinConcaveR"]
        ##分类

        output_cols_cat = ["ToolMaterial", "ToolType", "MachiningProcess", "MachiningMethod", "Axis",
                           "CutDirection", "Cutting_Sequence", "LayerToLayer", "EngageType", "ToolPosition"]
        # 数值

        output_cols_num = ["ToolDiameter", "LowerRadius", "ToolLength", "FluteLength",
                           "CuttingDepth", "PartStock", "FinalFloorStock", "Extension", "Tolerance", "SlopeAngle",
                           "EngageLength", "SpindleSpeed","FeedRates"]
        return input_cols_cat, input_cols_num, output_cols_cat, output_cols_num


def MachiningParameters(machiningfeaturevalue):
    if machiningfeaturevalue=='profile_planaring':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-planarMilling.csv'  # 修改为你的数据路径
        machiningfeaturevalue='profile_planaring'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\外形平面铣\Profile-planarMilling_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\外形平面铣\Profile-planarMilling_aux.pkl"
        return csv_path,machiningfeaturevalue,save_dir,model_save_pth,aux_save_path
    elif machiningfeaturevalue=='chamfer':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Chamfer-planarMilling.csv'
        machiningfeaturevalue='chamfer'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\倒角"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\倒角\chamfer_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\倒角\chamfer_weights.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path
    elif machiningfeaturevalue == 'endface':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\EndFace_FaceMilling.csv'
        machiningfeaturevalue='endface'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\端面"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\端面\machining_endface_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\端面\machining_endface_aul.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path
    elif machiningfeaturevalue == 'drillhole':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-hole-drill.csv'
        machiningfeaturevalue='drillhole'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\钻孔"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\钻孔\hole-drill_weights.pth"
        aux_save_path=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\钻孔\hole-drill_aux.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path
    elif machiningfeaturevalue == 'hole_holemilling':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Hole-HoleMilling.csv'
        machiningfeaturevalue='hole_holemilling'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\铣孔"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\铣孔\hole_holemilling_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\铣孔\hole_holemilling_aux.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path
    elif machiningfeaturevalue == 'pocket_planaring':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Pocket-PlanarMilling.csv'
        machiningfeaturevalue='pocket_planaring'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\平面铣型腔"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\平面铣型腔\pocket_planaring_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\平面铣型腔\pocket_planaring_weights.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path
    elif machiningfeaturevalue == 'profile_zlevel':
        csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-Zlevel.csv'
        machiningfeaturevalue='profile_zlevel'
        save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\等高铣外形"
        model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\等高铣外形\Profile-zlevel_weights.pth"
        aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\等高铣外形\Profile-zlevel_aux.pkl"
        return csv_path, machiningfeaturevalue, save_dir, model_save_pth, aux_save_path


# 1.加工 profile_planaring
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-planarMilling.csv'  # 修改为你的数据路径
# machiningfeaturevalue='profile_planaring'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\外形平面铣"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\外形平面铣\Profile-planarMilling_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\外形平面铣\Profile-planarMilling_aux.pkl"

# 2.加工倒角
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Chamfer-planarMilling.csv'
# machiningfeaturevalue='chamfer'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\倒角"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\倒角\chamfer_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\倒角\chamfer_weights.pkl"

# 3.加工端面
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\EndFace_FaceMilling.csv'
# machiningfeaturevalue='endface'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\端面"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\端面\machining_endface_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\端面\machining_endface_aul.pkl"

# 4.钻孔
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-hole-drill.csv'
# machiningfeaturevalue='drillhole'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\钻孔"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\钻孔\hole-drill_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\钻孔\hole-drill_aux.pkl"

# 5.铣孔
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Hole-HoleMilling.csv'
# machiningfeaturevalue='hole_holemilling'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\铣孔"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\铣孔\hole_holemilling_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\铣孔\hole_holemilling_aux.pkl"

# 6.平面铣型腔
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Pocket-PlanarMilling.csv'
# machiningfeaturevalue='pocket_planaring'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\平面铣型腔"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\平面铣型腔\pocket_planaring_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\平面铣型腔\pocket_planaring_weights.pkl"

# 7.等高铣外形
# csv_path = r'F:\DL-Study\MachiningParameterRecommend\data\train-Profile-Zlevel.csv'
# machiningfeaturevalue='profile_zlevel'
# save_dir = r"F:\DL-Study\MachiningParameterRecommend\损失及准确率\等高铣外形"
# model_save_pth=r"F:\DL-Study\MachiningParameterRecommend\eval_pth\等高铣外形\Profile-zlevel_weights.pth"
# aux_save_path="F:\DL-Study\MachiningParameterRecommend\eval_pth\平等高铣外形\Profile-zlevel_aux.pkl"