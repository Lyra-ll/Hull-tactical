# 1. 创建包含所有特征的主训练文件
python create_features.py --input train.csv --output train_final_features.csv
# 2. 创建包含所有特征的持有期验证文件
python create_features.py --input test.csv --output test_final_features.csv

# 1. 一次性生成所有特征文件 (新命令)
python create_features.py --train_input train_for_development.csv --test_input holdout_for_testing.csv --train_output train_final_features.csv --test_output test_final_features.csv

```3.  **检查**: 您的文件夹中现在应该有 `train_final_features.csv` 和 `train_val_final_features.csv` 这两个文件，它们包含了**原始特征 + 近500个手工特征 + 多目标列**。

流程1：特征海选
目的: 对所有原始特征进行重要性排序，为后续步骤筛选出“精英候选人”。
命令:
python main.py --mode rfe --config config

流程2：模型参数调优
目的: 基于上一阶段选出的Top N特征，为LGBM和AE模型寻找最优的超参数组合
# 1. 调优LGBM
python main.py --mode tune_lgbm --config config

# 2. 调优AE
python main.py --mode tune_ae --config config
产出: 生成 best_params_vXX_lgbm.json 和 best_params_vXX_ae.json 文件。

流程3：模型交叉验证
目的: 使用找到的最优特征和最优参数，进行一次完整的5折交叉验证，得到一个稳健的“模拟考”分数（CV AUC）。
命令:
python main.py --mode validate --config config
产出: 屏幕上会打印出最终的 OOF 平均 AUC 分数。

流程4：最终审判日
目的: 用全部开发数据训练最终模型，并在从未见过的“私藏”数据 (train_val.csv) 上进行评估，得到最真实的“高考”分数（Holdout AUC）
python main.py --mode holdout --config config
产出: 屏幕上会打印出最终的 持有集平均AUC分数。



#################################
快速代码验证 (冒烟测试)
# 使用smoke配置，只会跑3次试验，很快就能出结果
python main.py --mode tune_ae --config smoke

 # 快速测试RFE模式
    python main.py --mode rfe --config smoke
    
    # 检查：会生成一个名为 feature_ranking_smoke.csv 的文件

    # 快速测试LGBM调优
    python main.py --mode tune_lgbm --config smoke

    # 检查：会生成一个名为 best_params_smoke_lgbm.json 的文件

    # 快速测试AE调优
    python main.py --mode tune_ae --config smoke
    
    # 检查：会生成一个名为 best_params_smoke_ae.json 的文件
冒烟测试验证
python main.py --mode validate --config smoke
python main.py --mode holdout --config smoke
###############################################################################

Paul Fornia (24th Place) 方案知识点浓缩 (可用于README)
核心哲学
几乎完全忽略公开排行榜（Public LB），绝对信任本地交叉验证（CV）的分数 。
核心思路是通过各种工程技巧，在“分类”问题的框架下，尽可能多地融入“回归”问题的信息。
关键战术
动态标签平滑 (Dynamic Label Smoothing)
问题: 传统的0/1分类标签丢失了回报大小的信息。

方案: 使用 sigmoid(a * resp) 作为“软标签”进行训练 。

效果: 强制模型学习回报的“幅度”，对回报接近零的不确定性交易，输出接近0.5的模糊预测，提升了模型的稳健性 。
对数变换样本权重 (Log-Transforming Weights)
问题: 少数权重极大的样本会过度影响模型训练和CV评估。

方案: 在计算CV分数时，对样本权重 weight 进行对数变换 (log(weight)) 。

效果: 在保留重要样本影响力的同时，平滑了极端值，让CV分数更稳定、更可靠 。












