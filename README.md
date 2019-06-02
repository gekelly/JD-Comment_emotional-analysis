# JD Comment_emotional analysis




**京东评论文本挖掘（产品口碑分析）**

## **一、文本挖掘方向及基本思路**

**文本挖掘方向**： 用于分析京东用户对手机的观点、态度、情绪、立场以及其他主观感情的技术。

**文本挖掘基本思路：**

1、探索性分析：观测数据信息（含数据字段、数据缺失情况、样本分布情况等）

2、数据预处理：包括去除无效标签、编码转换、文档切分、基本纠错、去除空白、大小写统一、去标点符号、去停用词、保留特殊字符等。

3、文本分词及特征提取：jieba中文文本分词模型、文本特征转化未向量空间模型、海量稀疏特征做特征提取。

4、分类建模和效果评估：选择特定分类模型，建立模型并作效果评估和结论分析。

## 二、探索性分析

1、查看原始数据前4条数据情况

 ![1559191855770](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\1559191855770.png)

2、查看数据集记录数、维度、数据类型情况

数据集大小21*3637，时间字段为数值型需转化为日期型

![1559191875351](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\1559191875351.png)


3、文本评分分布情况

![1559192979039](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\1559192979039.png)

4、评论发布时间分布情况

![1559193016232](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\1559193016232.png)

5、评论长度与评分关系情况

![1559193101931](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\1559193101931.png)

## 三、文本预处理

**1、中文分词**：著名的nltk包对分词有良好的效果，劣势在于对中文不友好。对此选用jieba包进行处理。这里我们把文本通过空格形式分词。为了更好了解jieba中文分词详细用法，提供下友情链接做参考：https://github.com/fxsjy/jieba

`data1['seg_words'] = data1['content'].apply(lambda x: ' '.join(jieba.cut(x)))`

**2、去除停用词**：文本中存在：喂、啊和标点符号等无效词，直接应用于模型建立过程，建立的模型效果差。因此需停用掉这类词与标点符号。因为TF-IDF支撑停工词，因此勿需做额外处理。中文常见停用词表网上有几个版本。现附上个链接供参考：https://github.com/goto456/stopwords

**3、其他处理**：前面思路提及的预处理手段嘛。无明显的无效规则，且有些内容直接停用处理就好。就不需在这里进行。

**4、特征提取**：拆分好的文本不能直接扔去建模，还需要把文本转化为稀疏矩阵。主要手段嘛，词袋模型、TF-IDF等，这里主要采用TF-IDF方法。附上链接了解下TF-IDF基本原理：http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

```
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

tfv = TFIV(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = word_list)  

tfv.fit(terms)
X_all = tfv.transform(terms)
```

将文本进行特征表示后，还进行特征选择，选出较优的特征。这一步操作可以有效特省模型性能，改善模型。本次模型选用卡方检验选取100个特征。其他特征选择方法不展开讲述，此处搬运工附上链接供学习。

https://blog.csdn.net/Bryan__/article/details/51607215

```
#特征选择
from sklearn.feature_selection import SelectKBest, chi2
select_feature_model = SelectKBest(chi2, k=100)    ##卡方检验来选择100个最佳特征
X_all = select_feature_model.fit_transform(X_all, y)  #减少特征的数量，达到降维的效果，从而使模型的方法能力更强，降低过拟合的风险
```



## 四、模型建立与选取

使用机器学习去做情感分析。特征值是评论文本经过TF-IDF处理的向量，标签值评论分类为1（好评）、0（差评）。主要选取模型有：朴素贝叶斯、逻辑回归、SVM。对比下模型拟合效果。

### 1、朴素贝叶斯

```
model_NB = MNB()`
`model_NB.fit(x_train, y_train) #特征数据直接灌进来`
`MNB(alpha=1.0, class_prior=None, fit_prior=True) # ”alpha“是平滑参数，不需要掌握哈。`

`from sklearn.model_selection import cross_val_score`
`#评估预测性能，减少过拟合`
`print("贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(model_NB, x_train, y_train, cv=20, scoring='roc_auc')))` 
```

贝叶斯分类器20折交叉验证得分:  0.42522380291464257

### 2、逻辑回归

```
from sklearn.linear_model import LogisticRegression as LR`
`from sklearn.model_selection import GridSearchCV`

`model_LR = LogisticRegression(C=.01) # C是正则化系数。`
`model_LR.fit(x_train, y_train)`
```

20折交叉验证得分:  0.7847334991132702

### 3、SVM

```
from sklearn.svm import LinearSVC

model_SVM = LinearSVC(C=.01) # C是正则化系数。
model_SVM.fit(x_train, y_train)
```

20折交叉验证得分:  0.827218077723803

### 4、混淆矩阵评估分类模型预测效果

```
from sklearn.metrics import confusion_matrix
y_predict = model_SVM.predict(x_test)
cm = confusion_matrix(y_test, y_predict)
cm
```

```
array([[  0,  26],
       [  0, 884]], dtype=int64)
```

可以看出，负类的预测非常不准，准确预测为负类的0%，应该是由于数据不平衡导致的，模型的默认阈值为输出值的中位数。比如逻辑回归的输出范围为[0,1]，当某个样本的输出大于0.5就会被划分为正例，反之为反例。在数据的类别不平衡时，采用默认的分类阈值可能会导致输出全部为正例，产生虚假的高准确度，导致分类失败。 处理样本不均衡问题的方法，首先可以选择调整阈值，使得模型对于较少的类别更为敏感，或者选择合适的评估标准，比如ROC或者F1，而不是准确度（accuracy）。另外一种方法就是通过采样（sampling）来调整数据的不平衡。其中欠采样抛弃了大部分正例数据，从而弱化了其影响，可能会造成偏差很大的模型，同时，数据总是宝贵的，抛弃数据是很奢侈的。另外一种是过采样，下面我们就使用过采样方法来调整。

## 五、过采样后，进行建模

SMOTE（Synthetic minoritye over-sampling technique,SMOTE），是在局部区域通过K-近邻生成了新的反例。对SMOTE感兴趣的同学可以看下这篇文章<https://www.jianshu.com/p/ecbc924860af>

```
def sample_balabce(X, y):
​    from imblearn.over_sampling import SMOTE
​    model_smote = SMOTE()
​    x_smote_resamples, y_smote_resamples = model_smote.fit_sample(X, y)
​    return x_smote_resamples, y_smote_resamples

rex, rey = sample_balabce(X_all, y)
rex_train, rex_test, rey_train, rey_test = train_test_split(rex, rey, 
​                random_state=0, test_size=0.25)
```

1、使用过采样样本进行模型训练，并查看准确率：

20折交叉验证得分:  0.8555073492492292

2、查看此时的混淆矩阵：

array([[556, 322],
​       [141, 736]], dtype=int64)

总结：通过过采样方法后SVM得效果有所上升，负样本的识别率大幅度上升。

## 六、后续优化方向

1、采用更复杂的模型，如神经网络
2、模型调参
3、构建行业词典等









