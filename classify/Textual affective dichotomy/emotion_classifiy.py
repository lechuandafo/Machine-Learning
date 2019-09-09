# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:25:40 2019

@author: Yang Lechuan
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import logging
import gensim
import matplotlib.pyplot as plt
from nltk.text import TextCollection
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
#from textblob import TextBlob

def rev_cut(review):
    cut_file = './rev_cut.txt'
    with open(cut_file, 'w', encoding='utf-8') as f:
        for rev in review:
            rev_cut = " ".join(nltk.word_tokenize(rev))#对句子进行分词
            f.writelines(rev_cut +'\n')
    return cut_file

def word_vec(cut_file,model_name,dimension):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = gensim.models.word2vec.LineSentence(cut_file) #单词隔开，换行符换行
    model = gensim.models.word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=dimension) 
    model.save(model_name)
    
def get_sents_word_vec(review,dimension,model_name):
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    sent_vec = np.array([0]*dimension)
    sents_vec = []
    for rev in review:
        i = 0
        for word in nltk.word_tokenize(rev):
            sent_vec = sent_vec + model[word]
            i = i + 1
        sent_vec = sent_vec / float(i) 
        sents_vec.append(sent_vec)
    sents_vec = np.array(sents_vec)
    return sents_vec

def get_sents_word_vec2(review,num,dimension,model_name):
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    sents_vec = []
    for rev in review:
        i = 0
        sent_vec = []
        length = len(nltk.word_tokenize(rev))
        for i in range(0,num):     
            if (i<length):
                word = model[nltk.word_tokenize(rev)[i]]
            else:
                word = np.array([0]*dimension)
            sent_vec.append(word)
        sents_vec.append(sent_vec)
    sents_vec = np.array(sents_vec)
    sents_vec = sents_vec.reshape(len(review),num*dimension)
    #print(sents_vec.shape,sents_vec[0].shape)
    return sents_vec

def sent_vec(cut_file,model_name,dimension): #doc2vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences= gensim.models.doc2vec.TaggedLineDocument(cut_file)
    model = gensim.models.doc2vec.Doc2Vec(sentences,size=dimension, window=3)
    model.save(model_name)

def get_sents_sent_vec(review,dimension,model_name):#doc2vec
    sents_vec = []
    model = gensim.models.doc2vec.Doc2Vec.load(model_name)
    for i in range(0,len(review)):
        sents_vec.append(model[i])
    sents_vec = np.array(sents_vec)
    return sents_vec

def get_lab(labels):
    lab = []
    for label in labels:
        if(label == 'Negative'):
            lab.append([0])
        elif(label == 'Positive'):
            lab.append([1])
    lab = np.array(lab)
    return lab

def cal_auc(score,lab):
    threhold = 0
    dx = 0.005
    #dx = 0.1
    auc = 0
    pos_x = [] 
    pos_y = []
    while(threhold<=1):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        pred = []
        for i in range(0,len(score)):
            if(score[i] >= threhold):pred.append(1)
            else: pred.append(0)
        pred = np.array(pred)
        for i in range(0,len(lab)):
            if(lab[i] == 1): #正例
                if(pred[i] == 1): #真正例 
                    TP = TP + 1
                elif(pred[i] == 0):#假负例
                    FN = FN + 1
            elif(lab[i] == 0): #负例
                if(pred[i] == 1): #假正例
                    FP = FP + 1
                elif(pred[i] == 0): #真负例
                    TN = TN + 1
        FPR = FP/float(FP+TN)
        TPR = TP/float(TP+FN)
        pos_x.append(FPR)
        pos_y.append(TPR)
        threhold = threhold + dx
    x = np.array(pos_x)
    y = np.array(pos_y)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(x,y)
    auc = abs(np.trapz(y,x,dx=dx))
    return auc
    
def train_simplenn(train_data,sents_vec,test_sentsvec):
    labels = train_data['label']
    lab = get_lab(labels)
    dataset_size = sents_vec.shape[0]#样本数
    dimension = sents_vec.shape[1]#输入特征维度(特征个数)
    emb_unit_counts_1 = 25 #隐层结点数
    emb_unit_counts_2 = 25 #隐层结点数
    batch_size = 16 #定义训练数据batch的大小
    tf.reset_default_graph()
    w1 = tf.Variable(tf.random_normal([dimension,emb_unit_counts_1],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([emb_unit_counts_1,emb_unit_counts_2],stddev=1,seed=2))#初试化权重矩阵（生成相应大小的两个随机正态分布，均值为0，方差为1）
    w3 = tf.Variable(tf.random_normal([emb_unit_counts_2,1],stddev=1,seed=3))
    x = tf.placeholder(tf.float32,shape=(None,dimension),name='x-input') #在第一个维度上利用None，可以方便使用不大的batch大小
    y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input') #在第一个维度上利用None，可以方便使用不大的batch大小
    b1 = tf.Variable(tf.zeros([emb_unit_counts_1]))# 隐藏层的偏向bias 
    b2 = tf.Variable(tf.zeros([emb_unit_counts_2]))# 隐藏层的偏向bias 
    b3 = tf.Variable(tf.zeros([1]))# 输出层的偏向bias
    learning_rate = 0.0015
    #learning_rate = 0.01
    #定义前向传播过程
    a1 = tf.nn.softplus(tf.add(tf.matmul(x,w1),b1)) #计算隐层的值
    a2 = tf.nn.softplus(tf.add(tf.matmul(a1,w2),b2)) #计算隐层的值
    y = tf.nn.sigmoid(tf.add(tf.matmul(a2,w3),b3)) #计算输出值
    
    #定义损失函数和反向传播算法
    loss = tf.reduce_mean(tf.square(y_-y))
    #loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))#cross_entropy
    #learning_rate = 0.001 #tf.train.exponential_dacay(0.1 ,STEPS,dataset_size/batch_size , 0.96 , staircase = True)
    #optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)  #在此神经网络中只有基础学习率的设置，没有指数衰减率，也没有正则项和滑动平均模型。
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  #在此神经网络中只有基础学习率的设置，没有指数衰减率，也没有正则项和滑动平均模型。
    X = sents_vec
    Y = lab
    #以下创建一个session会话来运行TensorFlow程序
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op) #在此利用以上两行对其中涉及的参数进行统一初始化     
        STEPS = 100000#设定训练的轮数
        for i in range(STEPS):
            start = (i*batch_size) %dataset_size
            end = min(start+batch_size,dataset_size)#每次选取batch_size个样本进行训练
            sess.run(optimizer,feed_dict = {x:X[start:end],y_:Y[start:end]})#通过选取的样本训练神经网络并更新其中的参数
            if i%1000==0:
                score,loss_value = sess.run([y,loss],feed_dict={x:X,y_:Y})
                print("After%dtraining step(s),loss on all data is%g"%(i,loss_value))
        saver=tf.train.Saver(max_to_keep=1)
        saver.save(sess,'ckpt/emtion')#,global_step=i+1
        print("AUC: ",cal_auc(score,lab))
        X2 = test_sentsvec
        test_pred = sess.run(y,feed_dict = {x:X2})
        ID = []
        for s in range(1,len(X2)+1):
            ID.append(s)
        ID = np.array(ID)
        result = pd.DataFrame({'ID':ID.T,'Pred':test_pred.T[0]})
        result.to_csv("./result.csv",index = None)

def train_NB_onehot(train_data,test_data):
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    lab = get_lab(labels)
    fs_train = []
    for i in range(0,len(train_rev)):
        cut_rev = nltk.word_tokenize(train_rev[i])    
        fs_dict = {}
        for j in range(0,len(cut_rev)):
            fs_dict[cut_rev[j]] = True
        fs_train.append((fs_dict,int(lab[i])))
    fs_test = []
    for i in range(0,len(test_rev)):
        cut_rev = nltk.word_tokenize(test_rev[i])    
        fs_dict = {}
        for j in range(0,len(cut_rev)):
            fs_dict[cut_rev[j]] = True
        fs_test.append(fs_dict)
    classifier=nltk.NaiveBayesClassifier.train(fs_train)
    label = 1
    train_score = []
    test_score = []
    for i in range(0,len(fs_train)):
        dist = classifier.prob_classify(fs_train[i][0])
        train_score.append(dist.prob(label))
    train_score = np.array(train_score,dtype="float32")
    for i in range(0,len(fs_test)):
        dist = classifier.prob_classify(fs_test[i])
        test_score.append(dist.prob(label))
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score.T})
    result.to_csv("./result.csv",index = None)
    
def train_NB_tfidf_skl(train_data,test_rev,all_rev):   
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
   
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    #使用朴素贝叶斯进行训练
    mnb = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)  
    mnb.fit(train_X,labs)    # 利用训练数据对模型参数进行估计
    train_score = [pred[1]for pred in mnb.predict_proba(train_X)]
    test_score = [pred[1]for pred in mnb.predict_proba(test_X)]
    print ('The Accuracy of Naive Bayes Classifier is:', mnb.score(train_X,labs))
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_GNB_tfidf_skl(train_data,test_rev,all_rev):   
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
   
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    #使用高斯贝叶斯进行训练
    #mnb = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)  
    mnb = GaussianNB(priors=[0.625, 0.375])
    mnb.fit(train_X.toarray(),labs)    # 利用训练数据对模型参数进行估计
    train_score = [pred[1]for pred in mnb.predict_proba(train_X.toarray())]     # 对参数进行预测
    test_score = [pred[1]for pred in mnb.predict_proba(test_X.toarray())]
    #print(test_score)
    #4.获取结果报告
    print ('The Accuracy of GaussianNB Classifier is:', mnb.score(train_X.toarray(),labs))
    #print(test_score)
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_GNB_wordvec(train_data,train_sentsvec,test_sentsvec):
    labels = train_data['label']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    train_X = train_sentsvec
    test_X = test_sentsvec
    #使用高斯贝叶斯+词向量进行训练
    mnb = GaussianNB(priors=[0.625, 0.375]) 
    mnb.fit(train_X,labs)    # 利用训练数据对模型参数进行估计
    train_score = [pred[1]for pred in mnb.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in mnb.predict_proba(test_X)]
    print ('The Accuracy of GaussianNB Classifier is:', mnb.score(train_X,labs))
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)
    
def train_GNB_docvec(train_data,train_sentsvec,test_sentsvec):
    labels = train_data['label']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    train_X = train_sentsvec
    test_X = test_sentsvec
    #使用高斯贝叶斯+句向量进行训练
    mnb = GaussianNB(priors=[0.625, 0.375]) 
    mnb.fit(train_X,labs)    # 利用训练数据对模型参数进行估计
    train_score = [pred[1]for pred in mnb.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in mnb.predict_proba(test_X)]
    print ('The Accuracy of GaussianNB Classifier is:', mnb.score(train_X,labs))
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_LR_tfidf_skl(train_data,test_rev,all_rev):  
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    train_rev = train_rev
    test_rev = test_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    logreg = linear_model.LogisticRegression()
    logreg.fit(train_X, labs)
    train_score = [pred[1]for pred in logreg.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in logreg.predict_proba(test_X)]
    #print(test_score)
    #4.获取结果报告
    print ('The Accuracy of logistic regression Classifier is:', logreg.score(train_X,labs))
    #print(test_score)
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)
    
def train_SVM_tfidf_skl(train_data,test_rev,all_rev):   
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    train_rev = train_rev
    test_rev = test_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
   
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    #使用SVM进行训练 
    clf = SVC(kernel='sigmoid',probability = True)#调参
    clf.fit(train_X,labs)#训练
    train_score = [pred[1]for pred in clf.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in clf.predict_proba(test_X)]
    #print(test_score)
    #4.获取结果报告
    print ('The Accuracy of SVM Classifier is:', clf.score(train_X,labs))
    #print(test_score)
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_KNN_tfidf_skl(train_data,test_rev,all_rev):   
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    train_rev = train_rev
    test_rev = test_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
   
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    #使用SVM进行训练 
    clf = KNeighborsClassifier() #调参
    clf.fit(train_X,labs)#训练
    train_score = [pred[1]for pred in clf.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in clf.predict_proba(test_X)]
    #print(test_score)
    #4.获取结果报告
    print ('The Accuracy of KNN Classifier is:', clf.score(train_X,labs))
    #print(test_score)
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_RF_tfidf_skl(train_data,test_rev,all_rev):   
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    train_lab = get_lab(labels)
    labs = train_lab
    corpus = all_rev
    train_rev = train_rev
    test_rev = test_rev
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
   
    train_X = tfidf[0:len(train_rev)]
    test_X = tfidf[len(train_rev):len(train_rev)+len(test_rev)]
    #使用SVM进行训练 
    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train_X,labs)#训练
    train_score = [pred[1]for pred in clf.predict_proba(train_X)]     # 对参数进行预测
    test_score = [pred[1]for pred in clf.predict_proba(test_X)]
    #print(test_score)
    #4.获取结果报告
    print ('The Accuracy of RF Classifier is:', clf.score(train_X,labs))
    #print(test_score)
    train_score = np.array(train_score,dtype="float32")
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,train_lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score})
    result.to_csv("./result.csv",index = None)

def train_NB_tfidf_nltk(train_data,test_data,all_rev):   
    all_rev = [nltk.word_tokenize(rev) for rev in all_rev]
    corpus = TextCollection(all_rev)
    labels = train_data['label']
    train_rev = train_data['review']
    ID = test_data['ID']
    lab = get_lab(labels)
    fs_train = []
    print(train_rev[0])
    for i in range(0,len(train_rev)):
        cut_rev = nltk.word_tokenize(train_rev[i])    
        fs_dict = {}
        for j in range(0,len(cut_rev)):
            fs_dict[cut_rev[j]] = corpus.tf_idf(cut_rev[j],train_rev[i])
        fs_train.append((fs_dict,int(lab[i])))
    fs_test = []
    for i in range(0,len(test_rev)):
        cut_rev = nltk.word_tokenize(test_rev[i])    
        fs_dict = {}
        for j in range(0,len(cut_rev)):
            fs_dict[cut_rev[j]] = corpus.tf_idf(cut_rev[j],test_rev[i])
        fs_test.append(fs_dict)
    
    classifier=nltk.NaiveBayesClassifier.train(fs_train)
    label = 1
    train_score = []
    test_score = []
    for i in range(0,len(fs_train)):
        dist = classifier.prob_classify(fs_train[i][0])
        train_score.append(dist.prob(label))
    train_score = np.array(train_score,dtype="float32")
    for i in range(0,len(fs_test)):
        dist = classifier.prob_classify(fs_test[i])
        test_score.append(dist.prob(label))
    test_score = np.array(test_score,dtype="float32")
    print("AUC: ",cal_auc(train_score,lab))
    result = pd.DataFrame({'ID':ID.T,'Pred':test_score.T})
    result.to_csv("./result.csv",index = None)
    
def test(file,train_data):
    labels = train_data['label']
    lab = get_lab(labels)
    data = pd.read_csv(file)
    pred = data['Pred']
    print(pred,lab)
    print("AUC: ",cal_auc(pred,lab))

if __name__ == '__main__':
    train_data = pd.read_csv("./train.csv",lineterminator='\n')
    test_data = pd.read_csv("./20190513_test.csv")
    word_model_name = "./word_emo.model"
    sent_model_name = "./sent_emo.model"
    train_rev =  train_data['review']
    test_rev = test_data['review']
    all_rev = train_rev.append(test_rev)
    num = 16
    dimension = 32 
    train_size = len(train_rev)
    test_size = len(test_rev)
    cut_file = rev_cut(all_rev)
    #word2vec训练词向量,至少运行一次，若保留词向量可注释
    word_vec(cut_file,word_model_name,dimension)
    #doc2vec训练句向量，至少运行一次，若保留句向量可注释
    sent_vec(cut_file,sent_model_name,dimension)
    ####词向量，直接求平均
    train_sentsvec1 = get_sents_word_vec(train_rev,dimension,word_model_name)
    test_sentsvec1 = get_sents_word_vec(test_rev,dimension,word_model_name)
    ####词向量，合并
    train_sentsvec2 = get_sents_word_vec2(train_rev,num,dimension,word_model_name)
    test_sentsvec2 = get_sents_word_vec2(test_rev,num,dimension,word_model_name)
    ####文档向量
    train_sentsvec3 = get_sents_sent_vec(all_rev,dimension,sent_model_name)[0:train_size]
    test_sentsvec3 = get_sents_sent_vec(all_rev,dimension,sent_model_name)[train_size:train_size+test_size]

    ###############高斯贝叶斯+词/文档向量###############
    ###因为朴素贝叶斯要求特征矩阵非负，所以只能用高斯贝叶斯
    #train_GNB_wordvec(train_data,train_sentsvec1,test_sentsvec1) #0.61831131
    #train_GNB_docvec(train_data,train_sentsvec3,test_sentsvec3) #0.52883599
    #train_GNB_tfidf_skl(train_data,test_rev,all_rev) #0.70844343

    
    #效果较好
    ###############机器学习方法+TF-IDF############
    #train_NB_onehot(train_data,test_data) #0.81020273
    #train_NB_tfidf_nltk(train_data,test_data,all_rev) #朴素贝叶斯——NLTK 0.60013516
    train_NB_tfidf_skl(train_data,test_rev,all_rev) #朴素贝叶斯-SKL 0.85319362
    #train_SVM_tfidf_skl(train_data,test_rev,all_rev) #支持向量机 0.77722
    #train_LR_tfidf_skl(train_data,test_rev,all_rev) #逻辑回归 0.8514
    #train_RF_tfidf_skl(train_data,test_rev,all_rev) #随机森林 0.79925816
    #train_KNN_tfidf_skl(train_data,test_rev,all_rev) #K近邻 0.78078132
    
    #效果不佳0.62
    """
    ########前馈神经网络+doc2vec########
    train_simplenn(train_data,train_sentsvec3,test_sentsvec3)
    """
   
    #效果一般0.73
    ########前馈神经网络+word2vec########
    #train_simplenn(train_data,train_sentsvec1,test_sentsvec1)
    
    """
    #batch要大才有用(64)，效果不佳0.61 
    ######### 前馈神经网络+词向量拼接#############
    train_simplenn(train_data,train_sentsvec2,test_sentsvec2)
    """
    #test('train_Pred_gru.csv',train_data)