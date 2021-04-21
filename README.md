# Covid-Lockdown-Mental-Data-Analysis
# 疫情封锁对各年龄段人群焦虑抑郁症状的影响与预测识别

### 施工中

# 1	引言
## 1.1	研究背景及意义
&emsp;&emsp;新型冠状病毒(SARS-CoV-2)传播引发全球性肺炎疫情，全世界面临前所未有的公众健康危机。为阻止病毒进一步传播扩散、保障民众生命财产安全，世界各国根据自身防疫形势出台了相应的防疫政策，要求民众保持社交距离，尽量居家并减少不必要外出。疫情严重的国家更是采取冻结城市交通、关闭边境、取消国际航班等更严厉的封锁政策减少人员大规模流动。这些政策不仅限制了市场自由流动，还一定程度地扰乱了供应链的稳定秩序，影响了企业的正常运转。为了维持生存，他们不得不采取降薪裁员等措施降低运转支出。<br>
&emsp;&emsp;新冠疫情传播迅速范围广泛，几乎所有国家都不同程度地受到其带来的影响：社会失业率不断上涨；民众恐慌情绪不断蔓延，甚至出现盲目抢购商品的现象；生活防疫物资也出现供应缺口……。许多家庭不仅失去了稳定可靠的经济收入，还面临着新冠肺炎的感染风险，昂贵的新冠感染治疗费用成为压垮他们的最后一根稻草。<br>
&emsp;&emsp;封锁政策限制了民众出行：不能外出锻炼；不能走亲访友；不能参与聚会；甚至不能出门到岗工作。长时间的居家不断地压抑着人们的情绪，人际交流的减少也难免让人产生孤独感，而因为疫情失业而带来的经济困难也容易引发不安和焦虑情绪。有调查显示，疫情期间接受心理辅导的人数也比正常时期增加，焦虑症和抑郁症相关确诊人数也有所上升。<br>
&emsp;&emsp;这些现象都提示着我们不仅要关注新冠疫情带来的生命健康问题，也要注意疫情封锁引发的社会心理问题。<br>
&emsp;&emsp;通过分析疫情封锁与民众心理状态的影响因素并建立合适的预测模型，社区服务人员和管理者可以更好地分析居民在疫情居家期间的心理健康状况，以给予及时的社区指导，帮助其缓解相关焦虑抑郁情绪。通过模型提前识别有潜在焦虑与抑郁症状的人员，也更好地帮助他们了解自身的心理健康状态，也能够更及时地提供相关心理疏导服务和医疗保护。<br>

## 1.2	国内外发展状况
&emsp;&emsp;目前国内外已经注意到了疫情封锁对心理健康的影响，有部分学者认为这种影响甚至要大于新冠病毒给人体造成的生理健康威胁，心理影响的持续时间可能远远长于病毒在体内留存的时间，疫情改变了人们的生活方式，且这种改变需要很长的时间才能缓慢恢复。全球在致力于研发病毒疫苗的同时，也在同步研究影响心理健康的疫情相关因素，包括经济因素、社会因素、社交健康风险等。对相关心理数据进行机器学习建模是目前比较新兴热门的研究方法。<br>
&emsp;&emsp;疫情是全球性的，其对各地的影响具有普遍性和相似性。中国较早地控制了本地的疫情，而后疫情时期需要解决的不仅仅是经济衰退重启和医疗保健问题，更应该解决疫情给居民造成的心理障碍，这种的心理影响需要一个数据化工具来有效评估。通过对疫情期间的心理健康数据进行分析建模，我们能一定程度上了解居民在这段时间内的心理健康情况，从而可以给予及时的心理干预和相关医疗建议。<br>

## 1.3	本文内容
&emsp;&emsp;当前在处理统计分析及建模问题时，我们通常会倾向于从多元线性回归、Logistic回归、Poisson回归等可解释性强的一般线性模型开始，过渡到Lasso回归和岭回归(Ridge Regression)等广义线性模型，最后再到复杂度及灵活度更高的支持向量机(Support-vector machine)、随机森林(Random Forest）、Xgboost等机器学习算法。<br>
&emsp;&emsp;本文采用循序渐进的分析思路对数据进行建模，即数据处理——特征整理——建模对比——模型调优¬——结论总结。首先对波兰疫情封锁期间的心理学调查问卷数据进行清洗与整理并填充缺失值，再用各建模算法的默认参数对数据进行粗建模，筛选出较为优异的两到三个类别的机器学习算法作进一步的调优，最终选择模型得分最高的预测模型。<br>
&emsp;&emsp;由于是对精神评估问卷数据进行建模预测，且目标是对样本的心理健康评级作准确的分类，而相较于传统的线性回归模型，基于机器学习算法的分类器往往能有更加出色的分类性能，故本文将着重对比不同机器学习算法的分类效率，结合统计分析相关知识，筛选出最为有效的、能够准确地对广泛样本进行分类的预测模型，并筛选出与疫情心理健康相关的影响特征。<br>
&emsp;&emsp;本文第一章主要介绍课题相关研究背景意义及有关概念，紧接着第二章是详细描述了前期的数据处理过程和建模前的准备工作，第三章则是具体不同模型的构建与调试过程，最后的第四章为本课题的讨论与总结。<br>

## 1.4	相关概念
### 1.4.1  焦虑症与抑郁症
&emsp;&emsp;焦虑症(anxiety disorder)是一种心理障碍，亦或是一类精神疾病。其明显特征在于不自主不可控的焦虑和恐惧感。焦虑是对未来事件的担心，而恐惧则是对当前事件的反应[1]，这种障碍可能会引起心律不齐、震颤、惊恐发作、睡眠障碍等症状。因为其躯体症状较为普遍频繁且患者往往过度反应，所以时常与甲状腺功能亢进、心脏病等急性病症混淆。焦虑症包含一大类精神疾病，其中常见的有广泛性焦虑症(GAD)、社交焦虑症(SAD)、广场恐惧症、恐慌症、创伤后应激障碍(PTSD)和强迫症(OCD)等。多种焦虑症可以存在于同一位患者身上，且该症常与其他精神障碍同时发生，特别是重度抑郁，人格障碍等[2]，或者由其他精神障碍引起。焦虑症与正常的恐惧和焦虑情绪之间的不同之处在于过度的惊恐和焦虑以及持续时间长——一般超过六个月。<br>
&emsp;&emsp;抑郁症(Major depressive disorder，简称depression)是一种典型的精神障碍，其特征是情绪低落、缺乏自信、对正常有趣的事物和活动失去兴趣或经常出现未知病因的身体疼痛，且持续时间超过两周。它常发生于二三十岁的年轻人身上[3]，其中女性患病的概率大约是男性的两倍。医学界抑郁症的成因尚无明确界定，一般认为其成因是遗传因素[4]、心理因素与社会因素之间的相互作用，以及与后天的某些疾病如AIDS、哮喘等有关。生理水平上，抑郁症被认为与体内5-羟色胺(5-HT)水平有关[5]，这也被认为是常见抗抑郁药物的治疗原理。抑郁症和焦虑症一样，常与其他精神障碍共同出现，如双向情感障碍，又称躁狂抑郁症，且焦虑抑郁有非常强的关联性。<br>

### 1.4.2	 PHQ-ADS
&emsp;&emsp;PHQ-ADS指在评估时同时使用PHQ-9和GAD-7量表，广泛用于评估短期焦虑抑郁症状，为初步临床心理诊断提供参考。其中PHQ-9为患者健康问卷9项抑郁量表(Patient Health Questionnaire 9)，GAD-7则为7项广泛性焦虑症量表(Generalized Anxiety Disorder 7),是全球使用最广泛的抑郁和焦虑评估方法，共被翻译成100多种语言版本。<br>
&emsp;&emsp;PHQ-ADS共有16个问题(具体见附录A)，每个问题有四个选项，按程度依次记为0-3分，分值区间为0至48分。以分值10、20、30作为轻度、中度和重度焦虑抑郁障碍的评判阈值，可以由此判断测试者近期心理健康与躯体症状。Kroenke等人证明PHQ-ADS方法在临床实践中具有良好的灵敏度和准确率[6]，能够快速有效地区分患者焦虑抑郁程度。<br>

# 2	前期数据处理
## 2.1	数据基本描述
&emsp;&emsp;本次建模采用的原始数据来自2020年5月波兰进行的新冠疫情封锁有关心理学调查[7]，以PHQ-ADS标准综合评估测试者精神心理健康状态。该调查数据收集了1115个有效样本，每个样本对应包含了基本个人信息、PHQ-9与GAD-7量表详细得分与总分、疫情困难与社会支持量表对应得分(分别为16项和5项)、Covid-19可感知风险量表详细得分(6项)、个人健康问卷、经济情况等。具体的量表问题和数据说明可见于文末的附录A。<br>

## 2.2	数据缺失值处理
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

## 2.3	提取特征指标
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

# 3	模型构建与调优
&emsp;&emsp;本项目将对比常见的分类算法(决策树、KNN、SVM、随机森林、XGBoost)的数据拟合情况并测试其分类效能。鉴于目标变量Psy_Level具有3个类别标签，而常用于二分类评估的roc_auc得分函数不能很好地用于评估多分类模型，因此本次建模以f1_micro作为得分函数来评价模型。本文采用的建模思路为先使用各分类器默认参数的对数据进行大致的建模，对比相应的初步性能，再利用网格搜索对超参数作粗略调整，筛选出性能较为优秀的模型，最后对其作进一步的调优。<br>
&emsp;&emsp;在建模之前对数据做10折分层抽样，抽样比例为目标变量Psy_Level中三个标签的比例，并按7：3的比例将数据分裂为训练集和测试集，确保两个集中包含有同等比例标签的数据，减小划分比例对建模对比结果的影响。 网格搜索函数GridSearchCV会进一步用同样的分层抽样方法将训练集划分出一部分用于交叉验证模型参数的验证集。<br>

![image](https://github.com/Huang-Cc/Covid-Lockdown-Mental-Data-Analysis/blob/main/Images/pic.png)
> 图 1 默认参数下模型训练得分对比

&emsp;&emsp;从图 1可以看到，除决策树外，其余四类算法的F1_micro得分都大于0.7,其中得分最高的是随机森林为0.72836，SVM次之，但这四个分类器得分差距不大(<0.03)。鉴于上述对比仅是初步计算得分，建模时未对参数进行交叉验证，所以只表现的是模型性能的大致轮廓。<br>
&emsp;&emsp;下面将从决策树开始逐步对五种分类器进行初步调整，若得分有明显提升再对应进行颗粒度更小的参数搜索，并进行更深层的精细化调优。本实验将调用sklearn包modelselection类中的GridSearchCV函数对训练集数据进行10折交叉验证和参数网格搜索，以获得使得F1得分最高的参数组合。以下说明的模型参数均与sklearn包各模型算法一致。<br>













