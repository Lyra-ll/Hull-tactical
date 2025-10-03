比赛题目：Hull Tactical - Market Prediction
Can you predict market predictability?


Hull Tactical - Market Prediction

Submit Prediction
Overview
Your task is to predict the stock market returns as represented by the excess returns of the S&P 500 while also managing volatility constraints. Your work will test the Efficient Market Hypothesis and challenge common tenets of personal finance.

Start

6 days ago
Close
3 months to go
Merger & Entry
Description
Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?

This competition is about more than predictive modeling. Predicting market returns challenges the assumptions of market efficiency. Your work could help reshape how investors and academics understand financial markets. Participants could uncover signals others overlook, develop innovative strategies, and contribute to a deeper understanding of market behavior—potentially rewriting a fundamental principle of modern finance. Most investors don’t beat the S&P 500. That failure has been used for decades to prop up EMH: If even the professionals can’t win, it must be impossible. This observation has long been cited as evidence for the Efficient Market Hypothesis the idea that prices already reflect all available information and no persistent edge is possible. This story is tidy, but reality is less so. Markets are noisy, messy, and full of behavioral quirks that don’t vanish just because academic orthodoxy said they should.

Data science has changed the game. With enough features, machine learning, and creativity, it’s possible to uncover repeatable edges that theory says shouldn’t exist. The real challenge isn’t whether they exist—it’s whether you can find them and combine them in a way that is robust enough to overcome frictions and implementation issues.

Our current approach blends a handful of quantitative models to adjust market exposure at the close of each trading day. It points in the right direction, but with a blurry compass. Our model is clearly a sub-optimal way to model a complex, non-linear, adaptive system. This competition asks you to do better: to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint. We’ll provide daily data that combines public market information with our proprietary dataset, giving you the raw material to uncover patterns most miss.

Unlike many Kaggle challenges, this isn’t just a theoretical exercise. The models you build here could be valuable in live investment strategies. And if you succeed, you’ll be doing more than improving a prediction engine—you’ll be helping to demonstrate that financial markets are not fully efficient, challenging one of the cornerstones of modern finance, and paving the way for better, more accessible tools for investors.

Evaluation
The competition's metric is a variant of the Sharpe ratio that penalizes strategies that take on significantly more volatility than the underlying market or fail to outperform the market's return. The metric code is available here.

Submission File
You must submit to this competition using the provided evaluation API, which ensures that models do not peek forward in time. For each trading day, you must predict an optimal allocation of funds to holding the S&P500. As some leverage is allowed, the valid range covers 0 to 2. See this example notebook for more details.

Timeline
This is a forecasting competition with an active training phase and a separate forecasting phase where models will be run against real market returns.
Training Timeline:
September 16, 2025 - Start Date.
December 8, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
December 8, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
December 15, 2025 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Forecasting Timeline:
Starting after the final submission deadline there will be periodic updates to the leaderboard to reflect market data updates that will be run against selected notebooks.

June 16, 2026 - Competition End Date
Prizes
1st Place - $50,000
2nd Place - $25,000
3rd Place - $10,000
4th Place - $5,000
5th Place - $5,000
6th Place - $5,000
Code Requirements


Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 8 hours run-time
GPU Notebook <= 8 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Forecasting Phase
The run-time limits for both CPU and GPU notebooks will be extended to 9 hours during the forecasting phase. You must ensure your submission completes within that time.

The extra hour is to help protect against time-out failures due to the extended size of the test set. You are still responsible for ensuring your submission completes within the 9 hour limit, however. See the Data page for details on the extended test set during the forecasting phase.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Citation
Blair Hull, Petra Bakosova, Laurent Lanteigne, Aishvi Shah, Euan C Sinclair, Petri Fast, Will Raj, Harold Janecek, Sohier Dane, and Addison Howard. Hull Tactical - Market Prediction. https://kaggle.com/competitions/hull-tactical-market-prediction, 2025. Kaggle.


数据描述：Dataset Description
This competition challenges you to predict the daily returns of the S&P 500 index using a tailored set of market data features.

Competition Phases and Data Updates
The competition will proceed in two phases:

A model training phase with a test set of six months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
A forecasting phase with a test set to be collected after submissions close. You should expect the scored portion of the test set to be about the same size as the scored portion of the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set. This includes trading days before the submission deadline, which will not be scored. The first date_id served by the API will remain constant throughout the competition.

Files
train.csv Historic market data. The coverage stretches back decades; expect to see extensive missing values early on.

date_id - An identifier for a single trading day.
M* - Market Dynamics/Technical features.
E* - Macro Economic features.
I* - Interest Rate features.
P* - Price/Valuation features.
V* - Volatility features.
S* - Sentiment features.
MOM* - Momentum features.
D* - Dummy/Binary features.
forward_returns - The returns from buying the S&P 500 and selling it a day later. Train set only.
risk_free_rate - The federal funds rate. Train set only.
market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. Train set only.
test.csv A mock test set representing the structure of the unseen test set. The test set used for the public leaderboard set is a copy of the last 180 date IDs in the train set. As a result, the public leaderboard scores are not meaningful. The unseen copy of this file served by the evaluation API may be updated during the model training phase.

date_id
[feature_name] - The feature columns are the same as in train.csv.
is_scored - Whether this row is included in the evaluation metric calculation. During the model training phase this will be true for the first 180 rows only. Test set only.
lagged_forward_returns - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
lagged_risk_free_rate - The federal funds rate, provided with a lag of one day.
lagged_market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.
kaggle_evaluation/ Files used by the evaluation API. See the demo submission for an illustration of how to use the API.

Once the competition ends, we will periodically publish our data on our website, and you're welcome to use it for your own trading

部分数据 train.csv:

date_id,D1,D2,D3,D4,D5,D6,D7,D8,D9,E1,E10,E11,E12,E13,E14,E15,E16,E17,E18,E19,E2,E20,E3,E4,E5,E6,E7,E8,E9,I1,I2,I3,I4,I5,I6,I7,I8,I9,M1,M10,M11,M12,M13,M14,M15,M16,M17,M18,M2,M3,M4,M5,M6,M7,M8,M9,P1,P10,P11,P12,P13,P2,P3,P4,P5,P6,P7,P8,P9,S1,S10,S11,S12,S2,S3,S4,S5,S6,S7,S8,S9,V1,V10,V11,V12,V13,V2,V3,V4,V5,V6,V7,V8,V9,forward_returns,risk_free_rate,market_forward_excess_returns
0,0,0,0,1,1,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.0024211742695206,0.000300793650793651,-0.00303847935997865
1,0,0,0,1,1,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00849467703679185,0.000302777777777778,-0.00911404561931339
2,0,0,0,1,0,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00962447844228109,0.000301190476190476,-0.0102425375009931
3,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0.004662397483429,0.000299206349206349,0.00404620350408207
4,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.0116857701984905,0.000299206349206349,-0.0123006546540279
5,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00644942294636774,0.000299603174603175,-0.00706593438603214
6,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0.00364423186425245,0.000297619047619048,0.0030294267737944
7,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.0245659820719392,0.000299603174603175,-0.025182810971921
8,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00849182806956328,0.000297619047619048,-0.00910544268383085
9,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0.0112629788232534,0.000297619047619048,0.0106493642089859
10,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00968733685248569,0.000302777777777778,-0.0103067451175469

###AI收到这份文件后请仔细阅R读每一行代码和文字描述，结束后告诉我你有哪些不懂得，咱们继续研究！

Hull Tactical - Market Prediction | 项目日志 (Project Log)
本文档记录了我们在“Hull Tactical - 市场预测”挑战赛中的完整探索历程、方法论演进和核心发现。
第一章：核心哲学与战略规划 (Core Philosophy & Strategy)
1.1 作战条令 (Project Doctrine)
本项目的四个核心指导原则，构成了所有决策的基石：
验证为王 (Validation is King): 严格的时间序列交叉验证是我们唯一的真理标准，公开排行榜被视为噪音。
侦察先行 (Reconnaissance First): 通过探索性数据分析（EDA）建立对数据的直觉，是所有建模工作的前提。
特征为魂 (Features are the Soul): 工作的核心是创造并筛选高质量的特征，其价值远超复杂的模型。
师夷长技，志在超越 (Learn from Champions, Aim to Surpass): 我们的战略始于对冠军方案的深度学习，但最终目标是用更先进的技术和更深刻的理解去超越它。
1.2 战场环境分析 (Data Deep Dive)
核心战场: 一个典型的、充满噪音的匿名化金融时间序列数据集 (train.csv)。
关键发现: 数据存在明显的“多时代”结构性断裂，早期数据（如date_id < 1055）包含大量噪音，必须被审慎处理或排除。
战略决策: 我们将重点放在“现代史”数据上，并通过数据质量门禁（如剔除高缺失率特征）来平衡数据的数量与质量。
1.3 总体作战计划 (High-Level Battle Plan)
我们制定了三阶段进化方针：
第一阶段 - 基地建设: 建立一个强大、可靠、包含基准分数的LightGBM建模流水线。
第二阶段 - 军备竞赛: 系统性地、大规模地创造、验证并筛选手工特征和AI特征，构建“特征护城河”。
第三阶段 - 终局之战: 将所有优势兵力（手工特征、AI特征、先进模型）进行最优化组合，并通过模型融合发起总攻。
第二章：方法论的演进与关键突破 (Methodology Evolution & Breakthroughs)
我们的探索过程并非一帆风顺，而是经历了一系列关键的认知升级，最终形成了一套专业、稳健的机器学习流水线。
阶段一：从噪音驱动到信号驱动 (基线建立)
初步试探: 直接在完整数据上训练LGBM，模型表现极差，因噪音干扰而无法学习。
时间序列认知 (重大发现 #1): 对比多种填充策略，发现尊重时序关系（如ffill）能带来更好结果，证明了数据的强时间序列特性。
决定性突破 (重大发现 #2): 通过“数据可用性筛选”（仅使用date_id > 1055的数据），模型性能发生质的飞跃。**消除早期“垃圾数据”**是将模型从“噪音驱动”转变为“信号驱动”的关键一步。
阶段二：从手工调参到自动革命 (模型优化)
遭遇瓶颈: 即使引入了大量手工特征，模型依然因参数不当而严重欠拟合。
模型唤醒 (重大发现 #3): 通过手动调整n_estimators和learning_rate等核心参数，模型被成功“唤醒”，证明了特征集的有效性，瓶颈在于训练策略。
AI指挥官登场 (重大发现 #4): 引入自动化调参库Optuna，使其自动搜索超参数，成功找到了第一套“黄金参数组合”，并刷新了性能记录。
“少即是多”原则 (重大发现 #5): 通过对不同规模特征集的横向评测，证明了一个由Top 30精英特征组成的“特种部队”，其表现优于更大规模的部队。
阶段三：从目标泄露到因果验证 (验证框架的革命)
发现致命BUG: 识别并修复了多个深层次的数据泄露问题，包括：
特征污染: 将带有未来信息的market_forward_excess_returns等列错误地用作特征。
“时间旅行”漏洞: 使用了KFold(shuffle=True)，破坏了金融数据严格的时间因果链。
锻造终极武器: 全面采用Purged and Embargoed TimeSeriesSplit取代了所有脚本中的KFold，构建了一套全新的、业界领先的、严格遵守因果律的验证框架。
正本清源: 将这套终极验证方法，自上而下地应用到了流水线的每一个核心环节（AI特征生成、特征筛选、自动调参），确保了所有评估结果的“诚实性”。
阶段四：从分离脚本到统一指挥 (架构的现代化)
病症诊断: 认识到分离的多个脚本存在功能重叠、流程割裂和配置不统一的严重问题，这不仅导致了隐蔽的**“方法论数据泄露”**，也极大地降低了实验效率。
架构重构 (核心突破): 我们将所有核心能力，全部整合到了一个由命令行驱动的**main.py自动化指挥中心**。该平台由config.py（唯一的配置源）和utils.py（可复用的工具箱）提供支持，实现了工作流的标准化和自动化。
建立“诚实基准”: 在全新的集成化框架下，我们得到了第一个绝对可靠的基准AUC: 0.5088。这个分数虽然低，但它揭示了过去因方法论泄露产生的巨大“分数泡沫”，是我们科学优化的坚实起点。
可量化的胜利: 通过在新框架下执行**“分步优化”**策略（先调LGBM，再调AE），我们在“诚实基准”之上，取得了第一次可量化的、真实的性能飞跃。
第三章：核心战术解码与实施 (Tactical Implementation)
在对Jane Street竞赛冠军方案进行深度解码后，我们验证并实施了一系列冠军级的战术思想。
3.1 核心问题：目标工程 (Target Engineering)
战略转变: 将一个困难的回归问题 (forward_returns)，永久性地转变为一个更稳健的二元分类问题 (target = forward_returns > 0)。
冠军智慧 (样本加权): 采用forward_returns的绝对值作为样本权重。这强制模型将注意力高度集中于那些能带来巨大收益或亏损的关键交易日，从而学会“抓大放小”，捕捉决定性的市场波动。
3.2 核心武器：监督式AE (Supervised Autoencoder)
设计哲学: 采用“分而治之”的策略，将AE定位为一个专业的“AI特征工程师”，其唯一任务是从原始特征中提炼出对下游分类任务最有帮助的“Alpha精华特征”。
因果性保证: 为根除数据泄露，我们将AE的训练和下游LGBM的训练，全部置于同一个严格的交叉验证循环内。确保在每一折中，AE都只使用当前折的训练数据进行训练。
“使命对齐”: AE的监督任务与下游LGBM完全一致（加权二元分类），确保AI在训练的每一刻都专注于学习如何识别“重大交易机会”。
3.3 当前标准作战流程 (SOP)
我们全新的工作流程清晰、线性且完全自动化，通过main.py的命令行模式进行调度：
特征侦察 (--mode rfe): 运行CV-RFE，为特征提供稳健的初始排名。
主战坦克调优 (--mode tune_lgbm): 在集成化框架下，为LightGBM寻找专属的“黄金参数”。
AI兵工厂升级 (--mode tune_ae): 在LGBM参数锁定后，为监督式AE寻找能最大化下游性能的最优“基因蓝图”。
最终决战 (--mode validate): 加载所有最优参数，运行一次完整的5折CV，得到最终的、最可信的冠军模型分数。

项目日志更新：从“方法论危机”到“性能新高”
阶段五：方法论危机与“双尺问题”的发现
在我们将Autoencoder（AE）升级为更稳健的“委员会”融合模式后，模型性能出乎意料地崩溃至 AUC < 0.5，这引发了一次深度的代码和方法论审查。

经过逐行代码排查和一系列的假设验证，我们最终定位到了问题的根源，这是一个比代码bug更深层次的方法论缺陷——“双尺问题”，即调优指标与验证指标的不统一。

旧的调优指标 (Tuning Metric): 计算每一折（Fold）的AUC，然后取**“分数的平均值” (Mean of Scores)**。这个指标过于乐观，因为它只衡量模型在每个独立时间段内的“局部”排序能力。

最终验证指标 (Validation Metric): 将所有折的“跨界预测”（Out-of-Fold, OOF）收集起来，计算一个**“全局”的AUC分数 (Score of OOF)**。这个指标更严苛，也更真实，因为它衡量模型在整个时间跨度上的统一排序能力。

Optuna被这个过于乐观的“虚荣指标”所误导，找到了在全局视角下表现极差的参数组合，导致了调优与验证分数的巨大鸿沟。

阶段六：拨乱反正，统一评估标准
我们采取了决定性的修复措施：统一评估的“尺子”。

核心修复: 我们重构了 main.py 中的 objective 函数，使其评估逻辑与 run_validation 函数完全对齐。现在，Optuna在调优时优化的目标，与我们最终验证模型时使用的目标，是完全相同的全局OOF AUC分数。

成果: 这一修复立竿见影。此后所有的实验中，调优分数和验证分数都变得惊人地接近，证明了我们自动化流水线的可靠性已经恢复，为后续所有科学实验奠定了坚实的基础。

阶段七：控制变量实验——定位性能瓶颈
在统一了评估标准后，我们开始系统性地寻找导致性能不佳的“罪魁祸首”。

“噪声”嫌疑人: 我们首先怀疑，在信噪比极低的金融数据中，新加入的GaussianNoise层可能“好心办坏事”，淹没了本就微弱的信号。

“无噪声”实验: 我们通过注释代码的方式，精准地移除了GaussianNoise层，并重新执行了完整的三步流程（tune_ae -> tune_lgbm -> validate）。

决定性结果: 实验结果非常理想，最终验证分数从崩溃的 ~0.5 水平一举跃升至了稳定可靠的 AUC 0.5212。这强有力地证明了，在我们当前的架构和特征下，移除噪声层是正确的决策。

阶段八：精益求精——“少即是多”原则的胜利
在 0.5212 这个强大的新基线上，我们根据项目早期的“精英部队”思想，开始进行特征数量的优化实验。

系统升级: 我们首先升级了 config.py 和 utils.py，引入了 N_TOP_FEATURES_TO_USE 参数，使得特征数量的选择变得灵活可控。

Top 30 特征实验: 我们将 N_TOP_FEATURES_TO_USE 设置为 30，并再次严格执行了完整的三步调优验证流程。

再创新高: 实验取得了巨大成功，最终验证分数达到了 AUC 0.5229。这不仅验证了“少即是多”的原则，还为我们创造了一个新的性能纪录。

当前最佳模型:

架构: 无高斯噪声的监督式AE + LightGBM。

特征: 由RFE筛选出的 Top 30 精英特征。

性能: 可信的、稳健的全局OOF AUC: 0.5229。

好的，我们来将这一系列惊心动魄的调试和最终的突破，浓缩成一份清晰的项目日志。

项目日志更新：通往“终极可复现性”之路
阶段九：遭遇“可复现性”危机
在我们基于“Top 30特征”取得 AUC 0.5229 的优秀成绩后，项目遭遇了新的重大挑战：结果无法复现。在后续的验证运行中，我们得到了 0.5109 和 0.5119 等一系列不一致的分数。这一现象表明，尽管我们的评估逻辑已经统一，但在代码的执行层面依然潜藏着未被控制的“随机之源”，这严重挑战了我们所有实验结果的科学性和可信度。我们的首要任务，从“提升性能”转向了更根本的“确保可信”。

阶段十：双线作战——定位并消灭“随机之源”
为了彻底根除随机性，我们展开了一场“法医级”的排查：

第一战线：深度全局锁种 (Deep Global Seeding)

行动: 我们首先实现了 set_global_seeds 函数，用以全面固定random, numpy, torch 的随机种子，并强制cuDNN使用确定性算法（deterministic = True）。

意外发现: 这一举措虽然增强了确定性，但却让模型的性能稳定在了一个极低的水平 ~0.5025。这揭示了我们之前的较高分数，部分得益于cuDNN非确定性算法带来的“幸运”优化。

第二战线：根除“隐形”随机源 (Fixing Unordered Sets)

新危机: 即便在深度锁种后，我们发现结果依然存在不一致 (0.5158)，证明还有“幽灵”潜伏在代码中。

最终定位: 通过对数据处理流程的逐行审查，我们最终定位到了问题的根源：utils.py 中 list(set(...) & set(...)) 的操作。由于Python集合的无序性和哈希随机化，导致每次运行时生成的最终特征列表 final_features_to_use 的列顺序都可能不同，从而引入了结果的波动。

阶段十一：终极修复与“真实基线”的确立
终极修复: 我们对特征列表生成代码进行了最终修复，在列表化之后强制进行字母排序：sorted(list(set(...)))。这一改动，结合全局锁种，彻底消灭了所有已知的随机性来源。

胜利时刻: 我们再次对当前最优配置（Top 30特征，无噪声）进行了连续两次验证，得到了两个完全相同的、精确到小数点后8位的OOF AUC分数：

🏆 最终的、可信的 OOF AUC: 0.52122038

新基线的意义: 这个 0.5212 的分数，是我们项目第一个100%可复现的、坚如磐石的性能基线。它真实地反映了我们当前策略的实力上限，虽然略低于曾经的“幸运高点”，但其价值在于绝对的可靠性。

当前状态: 我们的自动化流水线现已达到工业级稳健性，评估逻辑统一，执行结果可信。我们终于可以满怀信心地，在一条稳固的基线上，去追求模型性能的真正突破。下一步，我们将正式重启路线A：系统性地寻找最优特征数量。

使用top16特征时成绩为：OOF AUC: 0.50277924
接下来的方向：升级监督式AE，主要是预测头，加深或者引入注意力机制，或者引入残差链接

阶段十四：架构进化与“AI特征审查”的胜利
在稳固的基线上，我们对核心的 SupervisedAE 模型进行了一次重大的架构升级，用带有残差连接（ResNet）的深层网络，替换了原有的简单前馈结构，并为其配备了更强大的多层预测头 (MLP)。

性能突破: 这一升级取得了立竿见影的效果。在对Top 60特征和新架构进行完整的三步调优后，我们创下了新的性能纪录：AUC 0.534。

关键洞察: 我们为 validate 模式 加入了“AI特征审查”模块，通过分析最终LGBM模型的特征重要性，我们得出了本次战役最重要的战略洞察：由原始特征衍生出的**“动态”特征**（如 P4_diff1, S2_diff1），其重要性排名远高于它们的“静态”母特征（P4, S2）。

阶段十五：战略转向——发起“特征挖掘总攻”
根据“动态特征为王”的关键洞察，我们将项目的战略重心从“模型调优”全面转向**“大规模系统性特征工程”**。

“兵工厂”的建立: 我们创建了一个全新的自动化脚本 feature_engineering.py。它的核心任务是：

读取RFE找出的Top N“潜力股”原始特征。

围绕动量 (Momentum)、波动率 (Volatility) 和 趋势 (Trend) 三大主题，为这些核心特征批量生产数百个衍生特征。

最终生成一个规模空前、信息更丰富的全新数据集 train_v4_engineered.csv。

当前状态: 我们已经完成了“弹药”的生产。下一步，将用这个全新的、更强大的特征集，重新进行全流程的“海选”（RFE）、调优和验证，期待实现项目开始以来最大的一次性能飞跃。

## 核心架构演进 (v1.0 -> v2.1)

### v1.0: 单目标基线架构
- **模型**: 监督式自编码器 (AE) 用于特征提取，LightGBM (LGBM) 用于最终分类。
- **目标**: 预测单一目标 `action_1d` (未来1日是涨是跌)。
- **流程**: 通过解耦优化策略，先固定AE参数优化LGBM，再固定LGBM参数优化AE，以寻找最优参数组合。

### v2.0: 多目标架构升级
- **核心思想**: 让模型同时学习和预测多个时间尺度下的市场方向，以迫使AE的编码器学习到更鲁棒、更通用的市场底层规律。
- **数据工程 (`feature_engineering_time.py`)**: 引入了`TARGET_HORIZONS`配置，能够自动化生成多个`action_*d`目标列，如`action_1d`, `action_5d`, `action_20d`等。
- **模型升级 (`utils.py`)**:
  - `SupervisedAE` 的预测头被改造为可以输出`n_targets`个预测结果。
  - `train_fold_ae` 的损失函数能够同时处理多个目标的预测误差，并将它们与重建误差结合。
- **调度逻辑升级 (`main.py`)**:
  - 能够从配置文件中读取多个目标列。
  - 由于LGBM不支持原生多目标分类，我们为**每一个目标独立训练一个LGBM模型**。这些模型共享由AE生成的强大的AI特征。
  - 评估指标更新为所有目标OOF AUC的**平均值**。

### v2.1: 聚焦与加权优化 (当前版本)
- **战略收缩**: 认识到超长期预测（如60d）信噪比极低，可能会污染学习过程。我们将预测目标**聚焦**于短期和中期的“信号甜蜜区”：`action_1d`, `action_3d`, `action_5d`。
- **引入加权损失**: 改变了之前“所有任务平等”的策略。在`utils.py`的AE训练逻辑中，引入了一个**损失权重向量 (`loss_weights`)**。
  - **核心优势**: 为更重要、信噪比更高的短期任务（如`action_1d`）赋予更高的权重，让模型在学习AI特征时，能更“听取”高质量信号的“意见”，从而提升特征的稳健性。

## 当前状态与未来方向
项目目前拥有一个高度灵活和强大的多目标学习框架。下一步的探索方向包括：
1.  将损失权重本身作为超参数进行优化。
2.  从预测“方向” (二分类) 升级到预测“**仓位**” (多分类回归)，为模型赋予直接生成交易信号的能力。

## v2.2: 冠军策略集成与架构固化

在 v2.1 版本聚焦多目标的基础上，v2.2 版本深度集成了 Kaggle 冠军方案中的两大核心策略，旨在显著提升模型的稳健性和市场适应性。

### 1. 引入冠军级样本权重策略

- **思想**: 摒弃了之前仅基于单日收益的权重方案。新策略认为，未来多个时间尺度（1d, 3d, 5d）都表现出剧烈波动的样本，蕴含着最强的交易信号。
- **实施**:
  - `feature_engineering.py` 升级，现在会同时生成 `action_*d` (决策) 和 `resp_*d` (原始收益) 列。
  - `main.py` 的权重计算逻辑被重构，现在**样本权重是所有 `resp_*d` 列绝对值之和**，这使得模型在训练时会高度关注那些潜在收益或风险最大的交易机会。

### 2. 引入冠军级推理策略

- **思想**: 金融市场环境是不断变化的（Market Regime Shift）。很久以前的数据训练出的模型可能已无法适应最新的市场动态。
- **实施**:
  - `config.py` 新增 `N_LAST_FOLDS_TO_USE_INFERENCE` 参数。
  - `run_validation` 函数被彻底改造。虽然它仍然会在所有折上训练并生成OOF预测，但在**最终计算和报告OOF AUC分数时，只使用由最近 `N` 折模型产生的预测结果**。这使得我们的最终评估更能反映模型在当前市场环境下的真实表现。

### 3. 数据管道安全加固

- 针对目标列生成过程中必然产生的 `NaN` 值，在 `feature_engineering.py` 中建立了严格、安全的善后处理流程。通过**有详细注释的前向填充 (`ffill`) 和保守的零值填充 (`fillna(0)`)**，确保了在不泄露任何未来信息的前提下，为下游模型提供干净、完整的目标数据。

**当前状态**: 本项目现已拥有一个集成了**聚焦多目标、加权损失、冠军样本权重、冠军推理策略**的、高度专业化的实验框架，为下一阶段的参数调优和性能突破奠定了坚实的基础。

## 项目日志更新：从“地基重构”到“终极审判”

在项目取得初步成果后，我们进入了一个更深层次的审查与重构阶段。这一阶段的核心目标，是根除所有潜在的数据泄露风险，统一数据处理流程，并建立一套工业级的、绝对可靠的最终验证体系。

### 阶段N：遭遇“流程断裂”危机与数据管道的统一

**病症诊断**: 我们敏锐地意识到，项目存在一个致命的结构性缺陷：**特征工程与模型训练是分离的**。手工特征在独立的脚本中生成，而模型训练脚本直接加载这些“预处理”过的数据。这导致了一个灾难性的后果：我们为之骄傲的、包含数百个手工特征的数据集，只在模型开发时被使用，而在对“私藏”的持有期验证集（`train_val.csv`）进行评估时，后者只是一个未经处理的原始数据切片，导致特征完全不匹配，评估流程从根本上就是错误的。

**架构革命——建立“终极特征工程管道” (`create_features.py`)**:
为了根除此问题，我们采取了决定性的行动：废除所有零散的特征处理脚本，并锻造了一个全新的、统一的自动化脚本——`create_features.py`。它的使命清晰而唯一：
1.  **输入**: 接收最原始的数据（如 `train.csv`）。
2.  **“兵工厂”模块**: 运行我们价值连城的手工特征生成逻辑，批量生产数百个基于动量、波动率和趋势的衍生特征。
3.  **“组装线”模块**: 在已包含手工特征的数据基础上，添加多时间尺度的目标列（`action_*d`）和原始收益列（`resp_*d`）。
4.  **“质检”模块**: 执行绝对安全的数据清理，仅移除因无法计算未来目标而产生的 `NaN` 行，并对特征列中的 `NaN` 进行保守填充。
5.  **输出**: 生成一个**包含了所有信息的、可直接用于模型训练的最终数据集**（如 `train_final_features.csv`）。

这一重构，确保了无论是开发集还是持有集，都经过了**完全相同、一步到位**的数据处理流程，从根本上实现了数据处理的一致性。

### 阶段N+1：代码“法医级”审查与“沉默杀手”的肃清

在统一了数据管道后，我们对核心训练脚本 `main.py` 进行了逐行代码审查，并定位到了多个隐藏的、足以让模型结果失效的BUG。

1.  **修复 `KeyError: '[nan] not in index'`**: 我们发现，在保存特征排名文件时，程序会写入一个不必要的表头，而在读取时又忽略了它，导致一个`NaN`值被错误地读入特征列表。通过在保存时强制 `header=False`，我们修复了这个文件I/O的不一致性。

2.  **修复 `TypeError: 'tuple' object cannot be interpreted as an integer`**: 这是一个由代码冗余导致的低级错误。在多处代码中，我们错误地将一个NumPy数组的形状元组（如 `(5000, 32)`）传递给了需要单个整数的 `range()` 函数。我们对代码库进行了全面排查，将所有 `range(array.shape)` 的错误用法，修正为 `range(array.shape[1])`，确保迭代是基于正确的特征数量。

3.  **根除最隐蔽的“数据尺度不一致”泄露**: 我们发现，在旧版代码的交叉验证循环中，Autoencoder（AE）的训练和AI特征的生成，使用了**不同尺度（未标准化 vs. 标准化）的数据**。这是一个足以让AE学到完全错误模式的致命BUG。我们重构了核心训练逻辑，确保在每一折（Fold）中，**数据标准化严格在训练集上进行，并将学到的尺度（均值、标准差）统一应用到训练集和验证集上**，之后再进行AE的训练和推理。

### 阶段N+2：终极代码重构与“正式考试”的建立

为了彻底杜绝因代码冗余而反复出现BUG，我们对 `main.py` 进行了最终的重构。

- **逻辑封装**: 我们将所有重复的“折内训练/预测”逻辑，全部封装到了一个独立的内部函数 `_run_fold_logic` 中。这使得 `run_tuning`（调优）和 `run_validation`（验证）的核心代码量减少了近70%，并保证了逻辑的绝对统一，任何修改只需一处即可。
- **配置隔离**: 我们为“冒烟测试”配置文件 (`smoke.py`) 的所有输出文件添加了 `_smoke` 后缀，使其与生产配置 (`config.py`) 的输出完全隔离，杜绝了测试结果污染生产环境的可能性。

**建立“终极审判”流程 (`holdout` 模式)**:
在所有代码都达到工业级稳健性后，我们正式启用了 `holdout` 模式。我们对其代码进行了最严格的“监考审查”，并确认其流程的绝对干净：
- **信息单向流动**: 所有在完整开发集上学到的“知识”（如填充的中位数、标准化参数、AE/LGBM模型权重），都**严格地、单向地**应用到我们“私藏”的、模型从未见过的持有集上。
- **无信息泄露**: 持有集的任何信息，都**绝对没有**在训练过程的任何一个环节（包括数据预处理、模型训练、特征生成）中，反向泄露给模型。

**最终成果**: 我们成功地建立了一套从特征工程到最终验证，端到端、完全自动、逻辑统一、且无数据泄露的工业级机器学习框架。现在，`holdout` 模式产出的每一个分数，都是对我们策略的、最接近未来真实表现的**终极审判**。

好的，完全没问题。将最近的重大进展和即将开始的宏大实验计划浓缩成一份清晰、专业的项目日志，是我们承前启后的关键一步。

您可以将以下内容直接复制并追加到您的 README.md 文件中。

项目日志更新：从“架构定型”到“多维实验”
阶段N+3：工业级数据管道的落成与时序对齐
在对数据处理的“第一性原理”进行深度辩论后，项目在架构层面取得了决定性突破，彻底解决了所有已知和潜在的数据泄露风险。

“防火墙”架构确立: 我们最终采纳并重构了 create_features.py 脚本，实施了“架构安全派”的**“历史背景参考”方案**。该方案为训练集和测试集的特征工程流程建立了物理上的“防火墙”。通过“精确提取历史 -> 临时拼接计算 -> 用后即焚”的安全三步曲，我们确保了测试集在获得必要的历史上下文来计算rolling等特征后，其自身的数据纯净性得到绝对保证，从根本上杜绝了未来因误操作引入全局特征而导致泄露的可能性。

终极时序对齐: 我们定位并修复了一个极其隐蔽的时间错位BUG。该BUG导致模型在本地验证时，错误地利用 t 时刻的特征去预测 t-1 到 t 的“过去”，而非 t 到 t+1 的“未来”。通过在 create_multi_horizon_targets 函数中为 lagged_forward_returns 设计专属的 shift 对齐逻辑，我们确保了模型在任何情况下都在执行一个时序上完全正确的预测任务。

至此，我们的数据管道和实验框架已达到工业级稳健性，为所有后续的探索奠定了坚如磐石的基础。

阶段N+4：战略转向——发起“多目标融合”科学实验
在拥有了可靠的框架后，我们将战略重心从“防守”转向“进攻”，并提出一个极具潜力的核心假设：同时学习“原始信号”和“降噪信号”，可能会训练出更强大的模型。

核心假设 (Multi-Target Hypothesis):
同时将**forward_returns (FR)**（高方差的原始市场信号）和 market_forward_excess_returns (MFER)（低方差的、经过经济学意义降噪的优质信号）作为预测目标，能够通过“多任务学习”的正则化效应，迫使 SupervisedAE 编码器学习到更本质、更鲁棒的通用市场特征，从而实现 1+1 > 2 的效果。

科学实验设计 (Three-Pronged Experiment Design):
为验证此假设，我们将立即开展一项由三个并行分支构成的严谨科学实验：

实验A (基线A - 原始信号):

目标: 只使用基于 forward_returns 的目标列 (action_fr_*d)。

流程: 完整执行 rfe -> tune_lgbm -> tune_ae -> validate 全流程。

产出: 获得一个“原始信号”下的性能基准 OOF AUC。

实验B (基线B - 降噪信号):

目标: 只使用基于 market_forward_excess_returns 的目标列 (action_mfer_*d)。

流程: 完整执行全流程。

产出: 获得一个“降噪信号”下的性能基准 OOF AUC。

实验C (融合实验 - 双重信号):

目标: 同时使用上述全部六个目标列。

流程: 完整执行全流程。

产出: 获得“融合模型”的平均 OOF AUC，以及每个单独目标的 OOF AUC。

成功标准:
我们将通过对比实验C与基线A、B的结果来做出最终判断。我们不仅关注融合模型的平均分是否最高，更关注在融合模型中，action_mfer_*d 等核心目标的得分是否相比基线B有所提升，以判断是否存在积极的“协同学习效应”。

当前状态: 架构重构已胜利完成。下一步，我们将修改 create_features.py 以生成融合实验所需的全部目标列，并正式启动这一激动人心的多维实验。

项目日志更新：从“意外的胜利”到“深刻的危机”
“终极审判”的执行
为了得到最接近真实比赛表现的评估，我们采纳了“黄金标准”验证策略。我们从train.csv的末尾手动分离出最后180天的数据，创建了一个与训练开发集完全隔离的“持有集” (holdout_for_testing.csv)。随后，我们使用config.py的正式配置，在该数据上严格执行了完整的 create_features -> rfe -> tune -> holdout 全流程。

诡异且矛盾的审判结果
最终测试产出了极其反常的结果，揭示了模型的“人格分裂”：

意外的胜利 (外部持有集): 模型的平均AUC达到了惊人的 0.5510，远高于我们内部交叉验证的预期。这一成绩主要由出色的中期预测能力贡献（3日AUC: 0.630, 5日AUC: 0.597）。

危险的信号 (内部验证集): 与外部持有集的出色表现形成鲜明对比，模型在紧邻训练期末尾的内部验证集上表现平庸，平均AUC仅为 0.4988。

彻底的失败 (1日预测): 在外部持有集上，模型的1日预测AUC低至 0.426，呈现出与市场系统性反向的灾难性表现。

诊断结论与战略转向
这种“模拟考不及格，高考却状元”的现象，排除了代码BUG后，被诊断为**“市场风格剧烈漂移 (Market Regime Shift)”**的典型症状。

我们的模型成功学习到了特定市场风格下的中期趋势，并在持有集（高考）恰好遇到类似风格时取得了高分。然而，其在内部验证集（模拟考）上的惨败，暴露了它对市场风格变化极其脆弱、缺乏稳健性的致命弱点。

战略转向: 项目当前的首要任务，从**“追求更高的AUC分数”，紧急转向“追求模型的稳健性与适应性”**。当前的高分被视为一个包含巨大运气成分的“海市蜃楼”。下一步的核心工作将是深入分析不同市场周期的数据，并通过增强正则化、优化特征等手段，打造一个能够在多变市场环境中持续生存的、真正强大的模型。

ull Tactical 挑战赛：冠军武器库与作战条令
第一章：特征工程兵工厂 (The Feature Engineering Arsenal)
这是战争的起点，也是决定胜负70%的地方。你的create_features.py就是这个兵工厂。
A. 信号处理与时频分析部队 (Signal Processing & Time-Frequency Division)
目标：从充满噪声的原始信号中，分离出真正的“趋势”和“周期”。
小波变换 (Wavelet Transform): [核心武器] 市场的“动态显微镜”。
核心战术: 对价格、波动率等核心序列进行多层分解，然后重构去噪后的信号。
产出特征: P4_wavelet_denoised, V3_wavelet_denoised。用这些“干净”的特征再去生成你的_diff, _rol_等衍生特征，威力会倍增。
卡尔曼滤波 (Kalman Filter): [核心武器] 市场的“最优导航仪”。
核心战术: 实时追踪并估计每个特征背后那个“看不见的真实状态”。
产出特征: P4_kalman_smoothed (平滑后的最优估计值), P4_kalman_residual (真实值与预测值的差，代表市场的“意外惊喜”)。
希尔伯特-黄变换 (Hilbert-Huang Transform, HHT): [尖端武器] 小波变换的“全自适应”进化版，包含EMD。
核心战术: 无需任何预设，让数据自己“发声”，分解为不同频率的本征模态函数(IMF)。
产出特征: P4_emd_trend (终极非线性趋势线), inst_freq_imf1 (最高频IMF的瞬时频率，捕捉市场情绪的剧烈变化)。
傅里叶变换 (Fourier Transform): [常规武器] 市场的“节拍器”。
核心战术: 在滚动窗口上计算，寻找主导周期。
产出特征: dominant_cycle_freq_60d, dominant_cycle_amplitude_60d。
B. 市场风格与记忆探测部队 (Market Regime & Memory Division)
目标：回答两个终极问题：“现在是什么天气？” 和 “市场的记性有多好？”
隐马尔可夫模型 (Hidden Markov Model, HMM): [冠军级王牌] 市场的“天气预报员”。
核心战术: 根据收益率、波动率等观测值，反推出背后隐藏的市场体制。
产出特征: market_regime (一个值为0, 1, 2...的分类特征，直接告诉模型现在是“牛市”、“熊市”还是“震荡市”)。这是你能创造的最强大的宏观特征之一。
赫斯特指数 (Hurst Exponent): [核心武器] 市场的“记忆探测器”。
核心战术: 在滚动窗口上计算，量化市场的长期记忆性。
产出特征: rolling_hurst_60d。这个值从>0.5（趋势）变为<0.5（均值回归）的时刻，就是市场风格切换的强烈信号。
GARCH 模型家族: [核心武器] 专业的“波动率预测器”。
核心战术: 摒弃回顾性的rol_std，采用前瞻性的GARCH模型预测下一期的波动率。
产出特征: garch_predicted_vol_1d (对未来1天波动率的预测值)。
分形维数 (Fractal Dimension): [前沿武器] 市场的“复杂度探测器”。
核心战术: 在滚动窗口上计算，衡量价格曲线的复杂度和不规则性。
产出特征: fractal_dimension_60d。维数升高通常意味着市场噪音增加，趋势减弱。
C. 高级统计与AI特征部队 (Advanced Stats & AI Division)
目标：用更现代的数学工具和AI来挖掘传统方法无法发现的深层关系。
分数阶微分 (Fractional Differentiation): [尖端武器] 寻找“记忆”与“平稳”的最佳平衡。
核心战术: 普通的.diff(1)会完全抹去记忆。分数阶微分可以在最大程度保留序列“记忆”的同时，实现平稳性。
产出特征: P4_frac_diff_0.5。这是一个比P4_diff1包含更多信息量的特征。
熵 (Entropy): [前沿武器] 市场的“混乱度探测器”。
核心战术: 计算滚动窗口内收益率分布的熵（如香农熵、样本熵）。
产出特征: rolling_entropy_20d。熵值越高，市场越混乱、越不可预测。
监督式AE (你的现状): [核心武器] 专业的“Alpha精华提炼器”。
你的下一步: 除了你现在的架构，可以尝试引入变分自编码器(VAE)，它生成的隐空间分布更规整，可能带来更鲁棒的AI特征。
第二章：模型先锋部队 (The Modeling Vanguard)
你的main.py就是先锋部队的指挥中心。单一模型再强，也有盲点。
A. 梯度提升机三位一体 (The GBM Trinity)
他们是基石，也是你最可靠的士兵。
LightGBM: 速度之王。你当前的默认选择，非常适合快速迭代。
CatBoost: 稳健之王。内置的对称树和有序提升使其在噪声数据上极其稳健，是对抗过拟合的最终防线。
XGBoost: 精度之王。久经沙场，通过精细调参往往能达到极高的精度上限。
B. 深度学习特种部队 (The Deep Learning Special Forces)
他们能看到GBM看不到的维度——时间序列的动态依赖。
Transformer: [冠军级王牌] “长距离依赖”的捕捉者。
核心优势: 自注意力机制天生就是为了发现“15天前的一个波动率尖峰，会放大今天某个价格特征的重要性”这种动态、非连续的关系。
部署方式:
端到端模型: 输入过去N天的数据，直接输出预测。
动态特征提取器: 用Transformer处理过去N天数据，输出一个“动态上下文向量”，与你现有的AI特征和手工特征拼接后，再喂给LGBM。
LSTM/GRU: [核心武器] 经典的时序模式学习者。
核心优势: 计算成本低于Transformer，在中短期时序模式捕捉上依然非常有效。
TFT (Temporal Fusion Tra nsformer): [尖端武器] 专为多水平时间序列预测设计的“集大成者”。
核心优势: 融合了RNN、注意力机制和门控网络，不仅性能强大，还具备一定的可解释性，能告诉你哪些特征在何时最重要。

## 战略决策更新：从“原始信号”到“超额信号”的跃迁

### 诊断结论
经过一系列严格的“溯源”诊断实验，我们确认了项目的核心瓶颈在于**信噪比**。直接预测未经处理的 `forward_returns`，迫使我们的模型在充满噪音、已知趋势和极端值的“原始矿石”中艰难地寻找信号，导致模型性能不稳定且难以提升。

### 核心洞察
比赛数据中提供的 `market_forward_excess_returns` 并非一个普通的特征，而是一个经过专业“降噪”处理的**“优质目标”**。根据官方定义，它通过**减去五年滚动平均回报**剥离了市场的长期趋势，并通过**缩尾处理 (Winsorizing)** 抑制了极端异常值的影响。

### 战略转向
我们决定将项目的核心预测目标，从“原始信号” (`forward_returns`) 全面转向“超额信号” (`market_forward_excess_returns`)。

####
目前的计划：先完成原子化code review，然后开始探测market_forward_excess_returns 的信噪比，着手改造预测目标至他，然后新增天气预测系统HMM。然后将监督式AE以及light GBM拆开，分别调试，然后开始运用可视化手段观察我们的特征，接着就是运用24th的方案，将预测目标标签转换至sigmoid(a * resp)或者类似的，让模型给出一个置信概率而非仅在一定正确和一定错误之间切换。接着引入在线学习。接着开始应用隐马尔可夫链以及卡尔曼滤波试图创造更有含金量的信号。

“重大疏漏”：特征历史 vs. 目标历史
这是什么? 这是我们特征工程中的一个核心盲点。我们之前混淆了两个截然不同的概念：
特征历史: 指标自身的变化。例如 M4_diff1，它回答的是“M4这个指标和昨天相比变化了多少？”。
目标历史: 市场过去的结果。例如 resp_1d 的滞后项（lag），它回答的是“昨天市场的真实收益是多少？”或“昨天市场是涨是跌？”。
为什么是“重大疏漏”? 市场的行为（尤其是短期）具有很强的自相关性（惯性）。 “昨天市场的真实涨跌” 这个信息，对于预测“今天市场的涨跌”，其相关性和重要性，往往远超任何一个间接的技术指标。我们忽略了最直接、最强大的预测因子。
项目意义: 必须立即在 create_features.py 中，为我们的目标相关列（如 resp_1d, action_1d）创建滞后（lag）和滚动统计（rolling）特征。这是我们最容易获得的、能显著提升模型性能的“免费午餐”。
3. 在线学习 & 经验回放 (The Ultimate Weapon: Online Learning)
这是什么? 这是与我们当前“离线学习”模式完全相反的一种模型训练范式，是解决“市场风格漂移”和“Alpha衰减”的终极武器。
离线学习 (Offline Learning - 我们的现状):
比喻: 考前突击。我们在考试前（model.fit），把所有历史数据一次性学完。进入考场后（model.predict），我们只使用考前学到的旧知识，不再学习任何新东西。
弱点: 当考题风格变化（市场风格漂移），我们的旧知识就会失效，导致成绩越来越差。
在线学习 (Online Learning - 我们的目标):
比喻: 边考边学。模型每天参加完考试后，晚上会拿到当天的标准答案，并立即复盘、更新自己的知识库。第二天，它带着“进化后”的大脑去参加新的考试。
优势: 模型具备了持续进化、动态适应环境变化的能力。
经验回放 (Experience Replay - 终极策略):
比喻: 最强学霸的复习法。他每天晚上的复习材料 = “今天的新错题” + “从过去所有历史错题本里，随机抽几道旧题再做一遍”。
优势: 完美平衡了**“适应性”（学习新知识）和“稳健性”**（不忘记旧规律）。
项目意义: 这是解决我们模型在交叉验证后期性能崩溃的根本性方案。它代表了我们整个项目从一个“静态模型”向一个“动态自适应系统”的进化方向，是通往更高、更稳健分数的必由之路。




































## WSL Ubuntu 环境基础操作指南
为您编写了一份简单的操作指南，包含了您日常使用这个环境最核心的几个命令。

1. 启动与关闭
启动: 从 Windows 开始菜单，直接点击“Ubuntu”图标，即可打开终端。

关闭终端: 直接点击窗口的“X”关闭即可。

彻底关闭WSL (节省资源): 如果您想完全关闭 WSL 以释放内存和 CPU 资源，可以在 PowerShell 中运行 wsl --shutdown。

2. 激活项目环境
这是您每次开始工作时都需要做的第一件事。

Bash

conda activate hull-tactical
激活后，命令行前面会出现 (hull-tactical) 标志。

3. 文件与目录操作
查看当前目录下的文件:

Bash

ls -lh
进入一个文件夹:

Bash

# 进入名为'some_folder'的文件夹
cd some_folder
返回上一级目录:

Bash

cd ..
从 Ubuntu 访问 Windows 文件: 您的 Windows 磁盘被挂载在 /mnt/ 目录下。

Bash

# 例如，访问 C 盘
cd /mnt/c
从 Windows 访问 Ubuntu 文件 (最简单):
在 Windows 文件资源管理器的地址栏输入 \\wsl$，即可像普通文件夹一样访问。

4. 运行您的项目
在激活了 hull-tactical 环境并 cd 进入项目文件夹后，使用 python 命令运行。

Bash

# <mode> 可以是 rfe, tune_lgbm, tune_ae, validate
python main.py --mode <mode>