
# 疫情封锁对各年龄段人群焦虑抑郁症状的影响与预测识别(Covid-Lockdown-Mental-Data-Analysis)
Transky Hwang in 2021.4.23

- [1 引言](#1-引言)
  * [1.1 研究背景及意义](#11-研究背景及意义)
  * [1.2 国内外发展状况](#12-国内外发展状况)
  * [1.3 本文内容](#13-本文内容)
  * [1.4 相关概念](#14-相关概念)
   + [1.4.1 焦虑症与抑郁症](#141-焦虑症与抑郁症)
   + [1.4.2 PHQ-ADS](#142-phq-ads)
- [2 前期数据处理](#2-前期数据处理)
  * [2.1 数据基本描述](#21-数据基本描述)
  * [2.2 数据缺失值处理](#22-数据缺失值处理)
  * [2.3 提取特征指标](#23-提取特征指标)
- [3 模型构建与调优](#3-模型构建与调优)
  * [3.1 决策树](#31-决策树)
  * [3.2 KNN](#32-KNN)
  * [3.3 SVM](#33-SVM)
  * [3.4 随机森林](#34-随机森林)
  * [3.5 Xboost](#35-Xboost)
  * [3.6 特征重要性](#36-特征重要性)
- [4 总结与展望](#4-总结与展望)
- [参考文献](#参考文献)
- [附录](#附录)
  * [附录A PHQ-ADS问卷](#附录A-PHQ-ADS问卷)
  * [附录B 疫情困难调查问卷](#附录B-疫情困难调查问卷)


# 1 引言
## 1.1 研究背景及意义
&emsp;&emsp;新型冠状病毒(SARS-CoV-2)传播引发全球性肺炎疫情，全世界面临前所未有的公众健康危机。为阻止病毒进一步传播扩散、保障民众生命财产安全，世界各国根据自身防疫形势出台了相应的防疫政策，要求民众保持社交距离，尽量居家并减少不必要外出。疫情严重的国家更是采取冻结城市交通、关闭边境、取消国际航班等更严厉的封锁政策减少人员大规模流动。这些政策不仅限制了市场自由流动，还一定程度地扰乱了供应链的稳定秩序，影响了企业的正常运转。为了维持生存，他们不得不采取降薪裁员等措施降低运转支出。<br>
&emsp;&emsp;新冠疫情传播迅速范围广泛，几乎所有国家都不同程度地受到其带来的影响：社会失业率不断上涨；民众恐慌情绪不断蔓延，甚至出现盲目抢购商品的现象；生活防疫物资也出现供应缺口……。许多家庭不仅失去了稳定可靠的经济收入，还面临着新冠肺炎的感染风险，昂贵的新冠感染治疗费用成为压垮他们的最后一根稻草。<br>
&emsp;&emsp;封锁政策限制了民众出行：不能外出锻炼；不能走亲访友；不能参与聚会；甚至不能出门到岗工作。长时间的居家不断地压抑着人们的情绪，人际交流的减少也难免让人产生孤独感，而因为疫情失业而带来的经济困难也容易引发不安和焦虑情绪。有调查显示，疫情期间接受心理辅导的人数也比正常时期增加，焦虑症和抑郁症相关确诊人数也有所上升。<br>
&emsp;&emsp;这些现象都提示着我们不仅要关注新冠疫情带来的生命健康问题，也要注意疫情封锁引发的社会心理问题。<br>
&emsp;&emsp;通过分析疫情封锁与民众心理状态的影响因素并建立合适的预测模型，社区服务人员和管理者可以更好地分析居民在疫情居家期间的心理健康状况，以给予及时的社区指导，帮助其缓解相关焦虑抑郁情绪。通过模型提前识别有潜在焦虑与抑郁症状的人员，也更好地帮助他们了解自身的心理健康状态，也能够更及时地提供相关心理疏导服务和医疗保护。<br>

## 1.2 国内外发展状况
&emsp;&emsp;目前国内外已经注意到了疫情封锁对心理健康的影响，有部分学者认为这种影响甚至要大于新冠病毒给人体造成的生理健康威胁，心理影响的持续时间可能远远长于病毒在体内留存的时间，疫情改变了人们的生活方式，且这种改变需要很长的时间才能缓慢恢复。全球在致力于研发病毒疫苗的同时，也在同步研究影响心理健康的疫情相关因素，包括经济因素、社会因素、社交健康风险等。对相关心理数据进行机器学习建模是目前比较新兴热门的研究方法。<br>
&emsp;&emsp;疫情是全球性的，其对各地的影响具有普遍性和相似性。中国较早地控制了本地的疫情，而后疫情时期需要解决的不仅仅是经济衰退重启和医疗保健问题，更应该解决疫情给居民造成的心理障碍，这种的心理影响需要一个数据化工具来有效评估。通过对疫情期间的心理健康数据进行分析建模，我们能一定程度上了解居民在这段时间内的心理健康情况，从而可以给予及时的心理干预和相关医疗建议。<br>

## 1.3 本文内容
&emsp;&emsp;当前在处理统计分析及建模问题时，我们通常会倾向于从多元线性回归、Logistic回归、Poisson回归等可解释性强的一般线性模型开始，过渡到Lasso回归和岭回归(Ridge Regression)等广义线性模型，最后再到复杂度及灵活度更高的支持向量机(Support-vector machine)、随机森林(Random Forest）、Xgboost等机器学习算法。<br>
&emsp;&emsp;本文采用循序渐进的分析思路对数据进行建模，即数据处理——特征整理——建模对比——模型调优¬——结论总结。首先对波兰疫情封锁期间的心理学调查问卷数据进行清洗与整理并填充缺失值，再用各建模算法的默认参数对数据进行粗建模，筛选出较为优异的两到三个类别的机器学习算法作进一步的调优，最终选择模型得分最高的预测模型。<br>
&emsp;&emsp;由于是对精神评估问卷数据进行建模预测，且目标是对样本的心理健康评级作准确的分类，而相较于传统的线性回归模型，基于机器学习算法的分类器往往能有更加出色的分类性能，故本文将着重对比不同机器学习算法的分类效率，结合统计分析相关知识，筛选出最为有效的、能够准确地对广泛样本进行分类的预测模型，并筛选出与疫情心理健康相关的影响特征。<br>
&emsp;&emsp;本文第一章主要介绍课题相关研究背景意义及有关概念，紧接着第二章是详细描述了前期的数据处理过程和建模前的准备工作，第三章则是具体不同模型的构建与调试过程，最后的第四章为本课题的讨论与总结。<br>

## 1.4 相关概念
### 1.4.1 焦虑症与抑郁症
&emsp;&emsp;焦虑症(anxiety disorder)是一种心理障碍，亦或是一类精神疾病。其明显特征在于不自主不可控的焦虑和恐惧感。焦虑是对未来事件的担心，而恐惧则是对当前事件的反应[1]，这种障碍可能会引起心律不齐、震颤、惊恐发作、睡眠障碍等症状。因为其躯体症状较为普遍频繁且患者往往过度反应，所以时常与甲状腺功能亢进、心脏病等急性病症混淆。焦虑症包含一大类精神疾病，其中常见的有广泛性焦虑症(GAD)、社交焦虑症(SAD)、广场恐惧症、恐慌症、创伤后应激障碍(PTSD)和强迫症(OCD)等。多种焦虑症可以存在于同一位患者身上，且该症常与其他精神障碍同时发生，特别是重度抑郁，人格障碍等[2]，或者由其他精神障碍引起。焦虑症与正常的恐惧和焦虑情绪之间的不同之处在于过度的惊恐和焦虑以及持续时间长——一般超过六个月。<br>
&emsp;&emsp;抑郁症(Major depressive disorder，简称depression)是一种典型的精神障碍，其特征是情绪低落、缺乏自信、对正常有趣的事物和活动失去兴趣或经常出现未知病因的身体疼痛，且持续时间超过两周。它常发生于二三十岁的年轻人身上[3]，其中女性患病的概率大约是男性的两倍。医学界抑郁症的成因尚无明确界定，一般认为其成因是遗传因素[4]、心理因素与社会因素之间的相互作用，以及与后天的某些疾病如AIDS、哮喘等有关。生理水平上，抑郁症被认为与体内5-羟色胺(5-HT)水平有关[5]，这也被认为是常见抗抑郁药物的治疗原理。抑郁症和焦虑症一样，常与其他精神障碍共同出现，如双向情感障碍，又称躁狂抑郁症，且焦虑抑郁有非常强的关联性。<br>

### 1.4.2 PHQ-ADS
&emsp;&emsp;PHQ-ADS指在评估时同时使用PHQ-9和GAD-7量表，广泛用于评估短期焦虑抑郁症状，为初步临床心理诊断提供参考。其中PHQ-9为患者健康问卷9项抑郁量表(Patient Health Questionnaire 9)，GAD-7则为7项广泛性焦虑症量表(Generalized Anxiety Disorder 7),是全球使用最广泛的抑郁和焦虑评估方法，共被翻译成100多种语言版本。<br>
&emsp;&emsp;PHQ-ADS共有16个问题(具体见附录A)，每个问题有四个选项，按程度依次记为0-3分，分值区间为0至48分。以分值10、20、30作为轻度、中度和重度焦虑抑郁障碍的评判阈值，可以由此判断测试者近期心理健康与躯体症状。Kroenke等人证明PHQ-ADS方法在临床实践中具有良好的灵敏度和准确率[6]，能够快速有效地区分患者焦虑抑郁程度。<br>

# 2 前期数据处理
## 2.1 数据基本描述
&emsp;&emsp;本次建模采用的原始数据来自2020年5月波兰进行的新冠疫情封锁有关心理学调查[7]，以PHQ-ADS标准综合评估测试者精神心理健康状态。该调查数据收集了1115个有效样本，每个样本对应包含了基本个人信息、PHQ-9与GAD-7量表详细得分与总分、疫情困难与社会支持量表对应得分(分别为16项和5项)、Covid-19可感知风险量表详细得分(6项)、个人健康问卷、经济情况等。具体的量表问题和数据说明可见于文末的附录A。<br>

## 2.2 数据缺失值处理
&emsp;&emsp;本研究采用的数据已经经过了初步筛选处理，并去除了离群数据。而对于问卷数据来说，研究者往往要关注的是数据的缺失情况及对应的处理方法。下表是原始数据中各变量的数据缺失数目。<br>

> 表 1 数据缺失情况

|&nbsp;|index|missNum|&nbsp;|index|missNum|
|:----:|:----:|:----:|:----:|:----:|:----:|
|0|Id|0|28|Phq9_2|0|
|1|Sex|0|29|Phq9_3|0|
|2|Age|0|30|Phq9_4|0|
|3|AgeGroup|0|31|Phq9_5|0|
|4|Education|0|32|Phq9_6|0|
|5|FinancialSituation_General|0|33|Phq9_7|0|
|6|FinancialSituation_Pandemic|0|34|Phq9_8|0|
|7|IncomeContinuity|391|35|Phq9_9|0|
|8|HealthStatus|0|36|Gad7_1|0|
|9|Unemployed|0|37|Gad7_2|0|
|10|Student|0|38|Gad7_3|0|
|11|Pandemic_Difficulties_1|0|39|Gad7_4|0|
|12|Pandemic_Difficulties_2|0|40|Gad7_5|0|
|13|Pandemic_Difficulties_3|0|41|Gad7_6|0|
|14|Pandemic_Difficulties_4|0|42|Gad7_7|0|
|15|Pandemic_Difficulties_5|0|43|Covid19_risk_1|0|
|16|Pandemic_Difficulties_6|0|44|Covid19_risk_2|0|
|17|Pandemic_Difficulties_7|0|45|Covid19_risk_3|0|
|18|Pandemic_Difficulties_8|0|46|Covid19_risk_4|0|
|19|Pandemic_Difficulties_9|0|47|Covid19_risk_5|0|
|20|Pandemic_Difficulties_10|0|48|Covid19_risk_6|0|
|21|Pandemic_Difficulties_11|0|49|SocialSupport_1|0|
|22|Pandemic_Difficulties_12|0|50|SocialSupport_2|0|
|23|Pandemic_Difficulties_13|0|51|SocialSupport_3|0|
|24|Pandemic_Difficulties_14|0|52|SocialSupport_4|0|
|25|Pandemic_Difficulties_15|0|53|SocialSupport_5|0|
|26|Pandemic_Difficulties_16|0|54|PHQ9_sum|0|
|27|Phq9_1|0|55|GAD7_sum|0|

&emsp;&emsp;表 1说明了实验数据的质量和完整性良好，总共包含有55个变量，且只有变量IncomeContinuity存在391个缺失值。变量IncomeContinuity代表的是收入连续性，该数据缺失意味着参与测试者可能无法确定自己当前或者是之后的收入能否连续，表明的对疫情封锁期间的经济状况的不确定。通过查看数据后我们发现，在变量IncomeContinuity存在缺失的样本里面对应Unemployed变量值均为0，即未失业。而根据实际情况，大部分未失业者大多数都能够获得基本薪酬，只是相较于疫情前可能有所减少。另一方面，含缺失值的样本数接近于原始数据样本的1/3，直接去除会减少相当一部分的数据量，因此本课题决定通过填补的方式修复缺失数据，将缺失值填充为1，以便后续建模。<br>

> 表 2 IncomeContinuity缺失行Unemployed数据描述

|index|count|mean|std|min|25%|50%|75%|max|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|IncomeContinuity_Umemployed|391|0|0|0|0|0|0|0|

## 2.3 提取特征指标
&emsp;&emsp;每个临床心理量表都包含有几个核心问题和辅助问题，这些题目间有一定的相关性，但往往着重针对的问题不同。临床心理学应用中需要综合多个精神心理量表得分和核心问题的作答情况来判断患者心理健康状态。在社科领域，问卷调查也是常见的信息采集手段。<br>
&emsp;&emsp;本文选用的调查数据采用PHQ-ADS方法对参与者进行心理健康评估，根据其得分评判标准及后续分类可操作性，根据量表总分重编码为3个心理等级，形成机器学习分类器的目标特征Psy_Level ：<br>
> 0：0-20分，心理状态较为健康，偶有轻度的身体或内心的不适，能够自我调节负面情绪；<br>
> 1：20-40分，中度焦虑抑郁状态，有明显的身体不适，内心感到焦虑不安和失落，做事逐渐缺乏热情；<br>
> 2：40-48分，重度焦虑抑郁状态，外界无法点燃自己活动的热情，几乎对任何事物缺乏兴趣，感到躯体和内心无比痛苦。<br>

&emsp;&emsp;对于疫情困难量表，它包含16个问题(详见附录B)，根据问题类型本文把它分为居家困扰(第1-5、8题)、社交限制(第6、7、9题)、防护措施和生活改变(第10-13题)、内心疑虑(第14-16题)，并取对应的平均得分，相应生成的特征变量为Home_Trouble、Social_Restrictions、Protection_and_Life_Change、Worry_and_Fear。<br>
&emsp;&emsp;除此之外，还需要对其他的信息问卷作汇总和特征提取，处理完成后去除了相应的原始变量和量表数据，具体操作如下表 3：<br>

> 表 3 变量处理与特征提取

|涉及变量|处理方法|
|:---:|:---:|
|Covid19_risk_**(**指1-6)|总分加和并取平均，生成变量Covid19riskmean|
|SocialSupport_**(**指1-5)|总分加和并取平均，生成变量SocialSupportmean|
|FinancialSituation_Pandemic (1),<br>FinancialSituation_General (2)|用(1)-(2)，生成变量FinancialSituation_Change|

&emsp;&emsp;为了方便后续建模，确保分类器正确运行，本项目考虑把所有表示类别的相关变量Sex,AgeGroup,Education,IncomeContinuity,HealthStatus,Unemployed,Student,Psy_Level数值类型改为Category。<br>
&emsp;&emsp;最终的数据集拥有14个特征变量，1个目标变量，共1115条数据。<br>

# 3 模型构建与调优
&emsp;&emsp;本项目将对比常见的分类算法(决策树、KNN、SVM、随机森林、XGBoost)的数据拟合情况并测试其分类效能。鉴于目标变量Psy_Level具有3个类别标签，而常用于二分类评估的roc_auc得分函数不能很好地用于评估多分类模型，因此本次建模以f1_micro作为得分函数来评价模型。本文采用的建模思路为先使用各分类器默认参数的对数据进行大致的建模，对比相应的初步性能，再利用网格搜索对超参数作粗略调整，筛选出性能较为优秀的模型，最后对其作进一步的调优。<br>
&emsp;&emsp;在建模之前对数据做10折分层抽样，抽样比例为目标变量Psy_Level中三个标签的比例，并按7：3的比例将数据分裂为训练集和测试集，确保两个集中包含有同等比例标签的数据，减小划分比例对建模对比结果的影响。 网格搜索函数GridSearchCV会进一步用同样的分层抽样方法将训练集划分出一部分用于交叉验证模型参数的验证集。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/pic.png)
> 图 1 默认参数下模型训练得分对比

&emsp;&emsp;从图 1可以看到，除决策树外，其余四类算法的F1_micro得分都大于0.7,其中得分最高的是随机森林为0.72836，SVM次之，但这四个分类器得分差距不大(<0.03)。鉴于上述对比仅是初步计算得分，建模时未对参数进行交叉验证，所以只表现的是模型性能的大致轮廓。<br>
&emsp;&emsp;下面将从决策树开始逐步对五种分类器进行初步调整，若得分有明显提升再对应进行颗粒度更小的参数搜索，并进行更深层的精细化调优。本实验将调用sklearn包modelselection类中的GridSearchCV函数对训练集数据进行10折交叉验证和参数网格搜索，以获得使得F1得分最高的参数组合。以下说明的模型参数均与sklearn包各模型算法一致。<br>

## 3.1 决策树
&emsp;&emsp;决策树算法的主要参数有特征选取方法criterion、树的最大深度max_depth、节点再划分最小样本数min_samples_split、叶子节点最小样本数min_samples_leaf等。其中criterion可以选择gini(基尼不纯度)和entropy(信息增益)。本课题在该参数上选用的是基尼不纯度，其计算方法如下公式(1)，criterion=gini对应的决策树算法为CART算法。<br>

![image](https://latex.codecogs.com/svg.image?I_%7Bg%7D(p)=%5Csum_%7Bi=1%7D%5E%7BJ%7D(p_%7Bi%7D%5Csum_%7Bk%5Cneq%20i%7Dp_%7Bk%7D)=%5Csum_%7Bi=1%7D%5E%7BJ%7Dp_%7Bi%7D(1-p_%7Bi%7D)=%5Csum_%7Bi=1%7D%5E%7BJ%7D(p_%7Bi%7D-p_%7Bi%7D%5E2)=1-%5Csum_%7Bi=1%7D%5E%7BJ%7Dp_%7Bi%7D%5E2)   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(1)

&emsp;&emsp;其中J为类别个数，i∈{1,2,…,J}，p_i为第i类标签在数据集中的占比。<br>
&emsp;&emsp;另外对该算法设置了参数max_features=sqrt，对变量min_samples_split, max_depth进行1网格搜索，搜索范围为2-30和1-20，步长均为1。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/dt.png)
> 图 2 通过决策树F1_micro得分筛选超参数组合(min_samples_split, max_depth)

&emsp;&emsp;通过对超参数组合(min_samples_split, max_depth)进行网格搜索，得到的最优组合为(2,2)，对应的训练集和测试集的F1_micro得分差距甚微，可以认为模型拟合充分，但该得分也只是仅仅与默认参数的随机森林齐平。鉴于后续研究中有性能更为优异的模型，故不对该决策树进行更深入的调参。<br>

## 3.2 KNN
&emsp;&emsp;K近邻算法(K-Nearest Neighbors algorithm)是一种用于分类和回归的非参数统计方法，该算法依赖于距离进行分类。其超参数为近邻数n_neighbor、近邻计算算法algorithm、权重函数weight等。在建模时，KNN算法默认的近邻数为5，现对其进行网格搜索，范围为\[\1,100)，观察模型得分是否有提升。搜索前，分类模型算法algorithm已设置为auto，即自动选择最优的近邻算法，其余超参数均为算法默认值。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/knn.png)
> 图 3 通过F1_micro得分筛选超参数n_neighbor

&emsp;&emsp;从图 3中可以看到，在参数n_neighbor不断增大的过程中，模型从欠拟合迅速提升到拟合状态，从曲线峰值往后模型得分便缓慢下滑至0.72上下波动，网格搜索出的最佳近邻个数为14，训练集得分为0.75385，测试集得分为0.73134，相较于图 1最初模型得分有所上升，略高于决策树。在更新超参数n_neighbor=14后尝试将参数weights由默认的uniform改为distance，给邻居的贡献增加权重，以便较近的邻居比较远的邻居对平均值的贡献更大。从得分结果来看，模型测试集得分没有提升。接着固定权重为distance不变，遍历超参数p得到图 4模型得分曲线。 <br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/knn3.png)
> 图 4 超参数p网格搜索过程

&emsp;&emsp;这一步的网格搜索结果为p=2，测试集得分与weights使用uniform时相同，训练集得分略微降低，最终的模型为加权14NN分类器。但对比其他机器学习算法，KNN算法在该数据集下的性能则稍显平庸。<br>

## 3.3 SVM
&emsp;&emsp;支持向量机(support vector machine，简称为SVM)是一种用于分类和回归分析的监督式学习模型与相关算法，由AT＆T贝尔实验室的Vladimir Vapnik及其同事开发[8]。SVM将训练样本映射到空间中的某个点，在高维或无限维空间中构建一个超平面或一组超平面，并最大程度地扩大两个类别之间的距离，然后将新样本映射到相同的空间，并根据它们落在空间的哪一侧来预测其属于哪一个类别，除此之外SVM巧妙地使用核方法来高效地执行非线性分类，将其输入隐式映射到高维特征空间。SVM算法一般用于分类、回归或异常值检测等任务。<br>
&emsp;&emsp;支持向量机的超参数随核函数的变化而相应不同，常用的有线性核函数Linear，对应超参数为C；高斯核函数rbf，对应超参数C和gamma；多项式核函数poly，对应超参数为C、gamma和degree；以及Sigmoid核函数，常用超参数与rbf相同。<br>
&emsp;&emsp;按SVM分类器建模常用流程，先选择合适的核函数，再调整对应超参数。对kernel参数备选组(poly,rbf,linear,sigmoid)进行10折网格搜索，其中性能最好的核函数为多项式核函数poly，默认参数下测试集F1_micro得分为0.7403，其分类效果明显优于决策树，也好于上述KNN算法。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/svc2.png)
> 图 5 多项式核SVM分类器参数C选择过程

&emsp;&emsp;对超参数C在\[\1,100)步长为1的范围进行搜索验证，当C=1(即默认值)时模型得分最优，模型测试集得分为0.7403，与第一步结果相同，接着搜索超参数degree的取值。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/svc3.png)
> 图 6 多项式核SVM分类器参数degree选择过程

&emsp;&emsp;从图 6可以看到，在degree=3时模型F1_score得分出现峰值，输入测试集得到测试得分为0.7403，与第一步模型得分相同，原因是超参数degree的默认值为3。<br>
&emsp;&emsp;通过上述三步参数选择，得到了研究至此暂时性能最优的分类模型，接下来看看采用集成学习算法的随机森林和Xgboost对比SVM性能是否有差距。<br>

## 3.4 随机森林
&emsp;&emsp;随机森林(random forest)一种用于分类、回归和聚类的集成学习算法，它包含多个决策树的分类结果，并且将子树输出的类别众数作为输出。随机森林纠正了决策树对其训练集过度拟合的问题，所以其表现通常优于决策树，但准确性低于梯度提升树。TK. Ho于1995年提出了随机决策森林这一概念[9]，而后Leo Breiman提出了更加合理的随机森林算法[10]。他介绍了一种使用类似于CART过程，结合随机节点优化和bagging来构建不相关树森林的方法，这种bagging方法在不增加偏差的情况下降低了方差，从而带来了更好的性能。此外他还创新地在构建森林时使用袋外误差来代替泛化误差，并提出可以通过随机森林来对变量的重要性进行排序。<br>
&emsp;&emsp;随机森林拥有决策树中所有超参数，用于控制每棵树的生成和剪枝，以及相应的集成框架参数如：n_estimators分类器数目、oob_score是否使用out-of-bagging数据计算袋外误差来评估模型、criterion树的特征评价标准等。随机森林算法超参数众多，一次性对所有参数进行网格搜索需要大量的计算时间，也不现实，因此采用贪心的边缘搜索方法获得局部最优的参数组合，以下是参数选择过程。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/rf.png)
> 图 7 随机森林分类器数目n_estimators与模型F1_micro得分的关系

&emsp;&emsp;在固定算法所有其他参数为默认值的情况下，n_estimators在\[\1,300)的范围内进行搜索验证。从图 7可以看到当分类器数目逐步增加到78时，模型得分达到最大为0.73974.继续往后增加树的棵数，模型得分趋于平稳，没有明显上升的趋势。根据奥卡姆剃刀原理，n_estimators取78为最佳的森林数目，此时的测试集得分为0.72239。接下来组合搜索超参数min_samples_split和max_depth，范围分布是\[\2,30)和\[\1,20)步长均为1。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/rf2.png)
> 图 8 不同参数组(min_samples_split,max_depth)对模型交叉验证分数的影响

&emsp;&emsp;输出的最优组合为max_depth=11,min_samples_split=20,此次参数选择使得随机森林的训练集得分提升到了0.75385，而测试集得分则是上升到了0.73731,较第一步有明显的性能增强。而参数min_samples_leaf与min_samples_split有一定关系，所有下面对参数组(min_samples_leaf,min_samples_split)，进行筛选与调整。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/rf3.png)
> 图 9 不同参数组(min_samples_leaf,min_samples_split)对模型交叉验证分数的影响

&emsp;&emsp;筛选出的最优参数组为(min_samples_leaf=1,min_samples_split=20)，因为数值1本身就是min_samples_leaf的默认参数，所以训练集与测试集分数没有发生改变。最后再尝试调整超参数max_features，看能否进一步改进模型。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/rf4.png)
> 图 10 随机森林参数max_features与模型F1_micro得分的关系

&emsp;&emsp;GridSearchCV最终选择选择了max_features=3，模型得分依旧没有发生改变，原因是前面建模是选择的max_features=auto，算法会根据经验默认为其填充特征数开方的取整值，即<br>

![image](https://latex.codecogs.com/svg.image?int(%5Csqrt%5B2%5D%7Bn%5C_%7B%7Dfeature%7D)=int(%5Csqrt%5B2%5D%7B14%7D)=3) &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(2)

&emsp;&emsp;至此，本研究建立了一个性能较为优异的随机森林，最后再来看看Xgboost的表现。<br>

## 3.5 Xgboost
&emsp;&emsp;Xgboost是陈天奇等人开发一个可扩展的梯度提升树系统[11]，它高效地实现了GBDT算法并对相应的算法和程序运行作出了许多改进，能够使用非常少的运算资源来快速准确地解决现实世界中各种规模的数据问题。Xgboost的超参数众多，能够应对各种数据的不规则性，有着无与伦比的可塑性，其高度复杂的算法亦有非凡出色的性能。它采用后剪枝的方法，先训练一颗完整的树，再从外向内剪枝，这样不容易陷入局部最优。<br>
&emsp;&emsp;Xgboost提供了三种参数用于模型设置：通用参数、Booster参数和学习目标参数。其中通用参数一般使用默认即可。Booster包含TreeBooster和LinearBooster。其中TreeBooster中的参数与决策树类似，通常要设置的有学习率learning_rate，与损失函数有关的gamma，树的最大深度max_depth，子节点最小样本权重min_child_weight，建树样本比例subsample，建树时特征采样比例colsample_bytree。LinearBooster的主要参数则为reg_lambda和reg_alpha。除此之外最重要的是学习目标参数，本课题属于多分类任务，所以将目标参数设置为multi:softmax，此外还需添加num_class=3;eval_metric=mlogloss。下面是具体的调参过程。<br>
&emsp;&emsp;和随机森林一样，模型的超参数过多，不能确保组合起来遍历搜索，所以采用相同的贪心算法来分组网格交叉验证调参。表 4为初始模型参数。<br>

> 表 4 Xgboost初步建模输入默认参数

|parameter|value|
|:----:|:----:|
|learning_rate|0.1|
|subsample|1|
|max_depth|6|
|gamma|0|
|colsample_bytree|1|
|min_child_weight|1|

&emsp;&emsp;本次建模将首先从n_estimators开始，寻找最佳的迭代次数。<br>
&emsp;&emsp;由图 11可以发现模型得分先是快速上升到一个极点，而后下降至波动平稳。极值点的n_estimators为33，测试集得分为0.6985，似乎得分并不是特别理想，接下来调整进一步超参数组合min_child_weight和max_depth，观察其对应模型得分变化。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb1.png)
> 图 11 Xgboost参数n_estimators与模型F1_micro得分的关系

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb2.png)
> 图 12 用Xgboost模型F1_micro得分筛选参数组(min_child_weight,max_depth)

&emsp;&emsp;由3Dsurface 图 12可以看到随着两个参数的不断增大，模型得分呈现下滑趋势，筛选器给出的极值点最佳参数组为(min_child_weight=2,max_depth=2)，其相应训练集预测得分为0.73461538，而测试集预测得分为0.73731343，相对于步骤1来说模型性能提升明显。下一步我们再尝试调整损失参数gamma。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb3.png)
> 图 13 Xgboost参数gamma与模型F1_micro得分的关系

&emsp;&emsp;固定前述两步得到的最佳超参数组合，对参数gamma进行调整。从上图可以看到，超参数gamma的得分曲线相对曲折，先后出现了5个极点，达到0.73846的得分，随后便迅速下降至0.727附近来回波动。为简便计算，取第一个极值点作为最佳gamma参数，其值为0.16，对应测试集得分为0.7403，性能已经基本与SVM的持平。Xgboost的众多可调整参数给予其更强的挖掘潜力，下面再尝试调整采样比例参数，适当降低基分类器间的相关性，看Xgboost是否会有更出色的表现。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb4.png)
> 图 14 用Xgboost模型F1_micro得分筛选参数组(subsample,colsample_bytree)

&emsp;&emsp;以0.05的颗粒度对参数组(subsample,colsample_bytree)进行网格搜索，取值区间为\[\0.5,1\]\。图 14可以清晰地看到顶点的得分值超过了0.75，参数搜索结果为(subsample=0.5,colsample_bytree=0.95)，测试集得分更是达到了0.76418，说明增加基分类器的多样性后模型的性能有了显著的提升。<br>
&emsp;&emsp;另外Xgboost在代价函数中加入了正则项用于控制集成模型的复杂度，也降低了模型方差，防止过拟合。下面将尝试调整正则化参数(reg_alpha,reg_lambda)。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb5.png)
> 图 15 用Xgboost模型F1_micro得分筛选参数组(reg_alpha,reg_lambda)

&emsp;&emsp;微平均F1得分峰值出现在坐标(reg_alpha,reg_lambda)=(0.65,0.7)处，对应的训练集得分为0.76026，测试集得分为0.75522。对比上一轮调参，测试集的分数略微降低，但从两集得分的差距来看，模型的拟合是更加充分的。<br>
&emsp;&emsp;最后的一步，我们来重新搜索一遍学习率参数，进一步将其颗粒度缩小，在[0.001,3]的范围内搜索最佳学习率。<br>
&emsp;&emsp;图 16的曲线已经相对明晰了，最初设置的学习率0.1就是使模型效能最好的值，因而得分并没有发生改变。用最终的模型对数据全集作预测，得到的F1_micro分数为0.76323，远远优于前述的所有建模算法。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/xgb6.png)
> 图 16 学习率learning_rate与模型F1_micro得分的关系

## 3.6 特征重要性
&emsp;&emsp;由于Xgboost是基于树模型的集成学习算法，能够自发地在学习过程中进行特征选择，从而精简了部分数据预处理操作。并且，它与其他树类算法一样能够在学习的过程中计算的特征重要性。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/imp.png)
> 图 17  Xgboost特征重要性可视化

&emsp;&emsp;图 17绘制了Xgboost模型中每个特征对分类的贡献程度。贡献最大的前五名依次是Social_Restrictions、Protection_and_Life_Change、Worry_and_Fear、Unemployed和AgeGroup，可见疫情期间不同年龄组、社交隔离、失业风险和生活剧变等对民众心理健康有显著影响。<br>

# 4 总结与展望
&emsp;&emsp;在本次疫情封锁焦虑抑郁影响与预测研究中，本课题对相关临床心理学问卷数据进行特征浓缩提取并对PHQ-ADS总分进行标签划分，再对其他学习问卷进行特征提取与清洗。通过尝试使用决策树、KNN、SVM、随机森林与Xgboost等机器学习算法对数据进行模型拟合并初步调优，我们最终确认采纳了性能较好可塑性强的集成学习算法Xgboost。对其深度调参后，模型性能远比决策树和KNN优异，也略好于SVM和同为集成学习算法的随机森林。最终的Xgboost参数组合于下表中展示。<br>

> 表 5  Xgboost最终参数

|parameter|value|parameter|value|
|:----:|:----:|:----:|:----:|
|learning_rate|0.1|num_class|3|
|eval_metric|mlogloss|max_depth|2|
|subsample|0.5|colsample_bytree|0.95|
|min_child_weight|2|n_estimators|33|
|gamma|0.16|reg_alpha|0.65|
|objective|multi:softmax|reg_lambda|0.7|

&emsp;&emsp;最终的模型能够使分类得分控制在0.763左右。由于模型的标签预测分类恒定落于3类标签之内，因此F1_micro得分将与准确度差别不大，即分类的正确率大于0.76。又因为Xgboost的灵活配置与高度可塑，我们还能对自定义模型的目标函数和评估函数，只要求函数二阶可导即可，这提供了全新的可操作维度。<br>
&emsp;&emsp;纵观全文，本文认为可以进一步提高模型的分类效能，其中最重要的数据质量与规模，它是一切建模的基础，搜集更多数量特征更加多元更接地气的数据能够更容易有针对性地对该心理健康问题进行建模。其次，在建模之前将调查的量表数据进行因子分析，对题目的特征进行筛选和因子旋转并通过特征重要性舍弃部分贡献较低的特征，从而提升模型的分析质量。考虑到现实中患有焦虑抑郁的人数与患病程度成反比，相应的数据标签也会因此而不平衡，进入影响到分类器的评价指标，解决方案是对标签比例大的数据进行欠采样和比例小的进行过采样来平衡标签比例，亦或者尽可能扩大数据的采集规模从根本上解决该问题。<br>

# 参考文献
[1]	American Psychiatric Association. Diagnostic and statistical manual of mental disorders (DSM-5®)[M]. Washington, D.C.: American Psychiatric Pub, 2013.189-195.<br>
[2]	Craske MG, Stein MB. Anxiety[J]. Lancet. 2016;388(10063):3048-3059.<br>
[3]	American Psychiatric Association. Diagnostic and statistical manual of mental disorders (DSM-5®)[M]. Washington DC: American Psychiatric Pub, 2013.160-168.<br>
[4]	Sullivan, Patrick F., Michael C. Neale, and Kenneth S. Kendler. Genetic epidemiology of major depression: review and meta-analysis[J]. American Journal of Psychiatry, 2000, 157(10):1552-1562.<br>
[5]	Blier, Pierre, and Mostafa El Mansari. Serotonin and beyond: therapeutics for major depression[J]. Philosophical Transactions of the Royal Society B: Biological Sciences, 2013, 368(1615): 20120536.<br>
[6]	Kroenke, Kurt, Fitsum Baye, and Spencer G. Lourens. Comparative validity and responsiveness of PHQ-ADS and other composite anxiety-depression measures[J]. Journal of affective disorders, 2019, 246: 437-443.<br>
[7]	Gambin, Małgorzata, et al. Generalized anxiety and depressive symptoms in various age groups during the COVID-19 lockdown in Poland. Specific predictors and differences in symptoms severity[J]. Comprehensive Psychiatry, 2021, 105: 152222.<br>
[8]	Cortes, Corinna, and Vladimir Vapnik. Support-vector networks[J]. Machine learning, 1995, 20(3): 273-297.<br>
[9]	Ho, Tin Kam. Random decision forests[A]. Proceedings of 3rd international conference on document analysis and recognition. Vol. 1[C]. NJ: IEEE, 1995.<br>
[10]	Breiman L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.<br>
[11]	Chen, Tianqi, and Carlos Guestrin. Xgboost: A scalable tree boosting system[A]. Balaji Krishnapuram, Mohak Shah. Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining[C]. NY: Association for Computing Machinery,2016. 785-794.<br>

# 附录
## 附录A PHQ-ADS问卷
&emsp;&emsp;PHQ-ADS问卷包含PHQ-9问卷和GAD-7问卷，以下是问卷完整问题内容：<br>
> 表 6 PHQ-9(病人健康状况问卷-9)

|No.|Questions(Over 2 weeks)|Score|
|:---:|:---|:---:|
|1|做任何事都觉得沉闷或者根本不想做任何事|&nbsp;&nbsp;|
|2|情绪低落、忧郁或绝望|&nbsp;|
|3|难于入睡、半夜会醒，或相反，睡觉时间过多|&nbsp;|
|4|觉得疲倦或没有精力|&nbsp;|
|5|胃口不好或饮食过量|&nbsp;|
|6|觉得自己做得不好、对自己失望或有负家人期望|&nbsp;|
|7|难于集中精神做事，例如看报纸或看电视|&nbsp;|
|8|其它人可能会注意到您在动或说话的时候比平时慢； 或者相反，您坐立不安，比起平时有多余的身体动作|&nbsp;|
|9|想到自己不如死了算了，或者有自残的念头|&nbsp;|

> 表 7 GAD-7(广泛性焦虑量表-7)

|No.|Questions(Over 2 weeks)|Score|
|:---:|:---|:---:|
|1|感觉紧张、焦虑或不安|&nbsp;|
|2|无法停止或控制担忧|&nbsp;|
|3|对各种事情担心太多|&nbsp;|
|4|难以放松|&nbsp;|
|5|坐立不安，以至于很难安静地坐下来|&nbsp;|
|6|变得容易生气或急躁|&nbsp;|
|7|感觉害怕，好像有可怕的事情要发生一样|&nbsp;|

## 附录B 疫情困难调查问卷
> 表 8 疫情困难调查问卷

|No.|Questions(Over 2 weeks)|Score|
|:---:|:---|:---:|
|1|家庭的人际关系不融洽(感觉应当顺从指令，争吵等)|&nbsp;|
|2|无法独处(没有他人陪伴)|&nbsp;|
|3|日常工作增加|&nbsp;|
|4|孤独感，有被抛弃的感觉。|&nbsp;|
|5|在限制措施中感到无所适从（例如：对能否做某件事感到彷徨）|&nbsp;|
|6|不能与心爱的人或朋友见面|&nbsp;|
|7|不能与一般人见面|&nbsp;|
|8|无聊、单调的生活|&nbsp;|
|9|行动和旅行的限制|&nbsp;|
|10|在公共场所需要戴口罩|&nbsp;|
|11|由于新的规定，购物方式发生了变化（例如，减少了商店的人数，需要在网上购物）|&nbsp;|
|12|觉得自己自由受到限制（如自作决定）|&nbsp;|
|13|生活方式需要改变|&nbsp;|
|14|与病毒传播有关的危机感和焦虑|&nbsp;|
|15|形势的不确定性和不可预测性|&nbsp;|
|16|对疫情的整体情况感到厌倦|&nbsp;|
