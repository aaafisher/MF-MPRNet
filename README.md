MF-MPRNet: A Multi-Task Multi-Frequency Learning Framework for Machining Parameter Recommendation in Computer-Aided Manufacturing

## 📖 Abstract

During CNC machining, the reasonable setting of machining parameters directly affects part quality, machining efficiency, and machining safety. However,
in existing Computer-Aided Manufacturing (CAM) systems, the setting of machining parameters still mainly relies on the experience of NC programmers, 
resulting in low programming efficiency and insufficient consistency in process planning. To address these issues, this paper proposes
a CAM-oriented multi-frequency multi-task machining parameter intelligent recommendation framework (MF-MPRNet), which aims to achieve 
automatic recommendation and configuration of machining parameters and further drive tool path generation. First, a machining parameter
dataset matched with CAM process parameters is constructed based on historical machining experience. Then, a multi-frequency learning mechanism
is introduced, in which learning modules with different update frequencies are set so that the model can extract the variation patterns between 
machining features and machining parameters at different time scales, where high-frequency learning focuses on capturing short-term local variations 
and low-frequency learning focuses on extracting long-term stable patterns. On this basis, a multi-task learning mechanism is adopted to jointly predict
discrete parameters such as the machining process and continuous parameters such as cutting depth, feed rate, and machining allowance, achieving a complete
recommendation of machining parameters. The model takes machining features and their geometric attributes as input, and passes the prediction results to the
CAM system for automatic parameter configuration and tool path generation. Experimental results show that MF-MPRNet achieves an accuracy of 85.70% on the 
classification parameter prediction task and an R² of 0.8359 on the numerical parameter prediction task. Furthermore, physical machining experiments were 
conducted on an actual machine tool for validation. The toolpaths generated based on the recommended parameters successfully machined the target features, 
and the machined features were consistent with the design model, with critical dimensions achieving the IT8 tolerance grade.
 The results indicate that the proposed method can effectively achieve intelligent recommendation and automatic configuration of machining parameters
 on the CAM side, providing a feasible solution for the automation and intelligence of CNC machining process planning.



## 🛠️ Requirements
conda env create -f environment.yml

Train:
  run train.py

Ppredict:
  run predict.py
