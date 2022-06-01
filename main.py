import pandas as pd
from sqlalchemy import null
#from GANomaly import GANomaly
from GANomaly_small import GANomaly
from GANtrainer import GANtrainer
from LoadData import LoadData
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
import imageio
import glob
import tensorflow as tf
import os
from TimeSeriesGenerator.main import dataPreprocess_Main
import scipy.stats as st
import xlsxwriter

import random

from SVMcalssify import SVM_classifier
    
def train(x_ok, gan_trainer, g_e, g, e, f_e, d, cp, cpdir, classType, bz=32, epoch=1000, score_rate = 0.8, fig_interval=50):
    train_data_generator = get_data_generator(x_ok, bz)
    for i in range(epoch):
        if i==0:
            evaluate_fig(g, g_e, i+1, classType)
        
        x, y= train_data_generator.__next__()
        
        ### train disciminator ###
        d.trainable = True
        fake_x = g.predict(x)
        
        d_x = np.concatenate([x, fake_x], axis = 0)
        d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)
        
        d_loss = d.train_on_batch(d_x, d_y)
        
        ### train generator ###
        d.trainable = False        
        g_loss = gan_trainer.train_on_batch(x, y)
        
        if (i+1) % fig_interval == 0:
            print(f'epoch: {i+1}, g_loss: {g_loss}, d_loss: {d_loss}')
            save_checkpoints(cp, cpdir)
            evaluate_fig(g, g_e, i+1, classType, score_rate=SCORE_RATE)
            
def save_checkpoints(cp ,cpdir):
    cp.save(file_prefix = os.path.join(cpdir, "ckpt"))
    
def restore_checkpoint(cp, cpdir, target = 'latest'):
    if target == 'latest':
        cp.restore(tf.train.latest_checkpoint(cpdir)).expect_partial()
    else:
        cp.restore(target).expect_partial()
    
def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        indexes = np.arange(datalen)
        np.random.shuffle(indexes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, indexes[i*batch_size: (i+1)*batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]
            
def Calculate_Score(rate, x_test, g, g_e):
    #latent loss (Z - Z~)^2
    encoded = g_e.predict(x_test) #Ge(X)
    gan_x = g.predict(x_test) #X~ = G(Z)
    encoded_gan = g_e.predict(gan_x) #Ge( X~ )
    #sum_over_row = np.mean(np.abs(x_test - gan_x).reshape(-1,1024), axis=-1)
    #print(sum_over_row.shape)
    Apparent_loss = np.mean(np.abs(x_test - gan_x).reshape(-1,1024), axis=-1) #apparent loss = sum of (X - X~)
    #print(sum_over_row.shape)
    Latent_loss = np.mean(np.abs(encoded - encoded_gan), axis = -1) #latent loss = sum of (Ge(X) - Ge(X~))^2
    score = Apparent_loss * rate + (1.0-rate)*Latent_loss
    score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    return score
    
            
def evaluate_fig(g, g_e, epoch, classType, score_rate = 0.8):
    normal, abnormal = classType[0], classType[1]
    
    '''
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    score = np.sum(np.abs(encoded - encoded_gan), axis = -1)
    score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    '''
    score = Calculate_Score(score_rate, x_test, g, g_e)
    
    ''''''
    normal_score, abnormal_score = [], []
    for i in range(len(y_test)):
        if y_test[i]==1: normal_score.append(score[i])
        else: abnormal_score.append(score[i])
    normal_score, abnormal_score = np.array(normal_score), np.array(abnormal_score)
    n_max, abn_min = max(normal_score), min(abnormal_score)
  
    rcParams['figure.figsize'] = 14, 5
    rcParams['lines.markersize'] = 3
    plt.xlabel('testing data')
    plt.ylabel('Score')
    plt.scatter(range(len(x_test)), score, c=['skyblue' if y == normal else 'pink' for y in y_test])
    plt.title('epoch: {:04d}'.format(epoch))
    
    ''''''
    plt.plot([0, len(y_test)-1], [n_max,n_max],'k--', label = 'Normal Max: {0}'.format(n_max))
    plt.plot([0, len(y_test)-1], [abn_min,abn_min],'r--', label = 'Abnormal Min: {0}'.format(abn_min))
    plt.legend()
    
    #plt.show()
    # save plot to file
    filename = 'evaluate_results/scatter_plot_{:04d}.png'.format(epoch)
    plt.savefig(filename)
    plt.close()
    
def final_evaluate( g ,g_e, normal, confidence_rate = 0.99, score_rate = 0.8):  
    
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    #score = np.sum(np.abs(encoded - encoded_gan), axis = -1)
    #score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    
    
    score = Calculate_Score(score_rate, x_test, g, g_e)
    
    normal_score, abnormal_score = [], []
    for i in range(len(y_test)):
        if y_test[i]==1: normal_score.append(score[i])
        else: abnormal_score.append(score[i])
    
    normal_score, abnormal_score = np.array(normal_score), np.array(abnormal_score)
    (n_Tmin, n_Tmax) = st.t.interval(alpha=confidence_rate, df=len(normal_score)-1, loc=np.mean(normal_score), scale=st.sem(normal_score))
    (abn_Tmin, abn_Tmax) = st.t.interval(alpha=confidence_rate, df=len(abnormal_score)-1, loc=np.mean(abnormal_score), scale=st.sem(abnormal_score))
    n_max, abn_min = max(normal_score), min(abnormal_score)
    print('Interval of normal data in {0}% confidence: ({1},{2}), Max of Normal Score: {3}'
          .format(confidence_rate*100.0, n_Tmin,n_Tmax, n_max))
    print('Interval of abnormal data in {0}% confidence: ({1},{2}), Min of Abnormal Score: {3}'
          .format(confidence_rate*100.0, abn_Tmin,abn_Tmax, abn_min))
    
    rcParams['figure.figsize'] = 28, 10
    rcParams['lines.markersize'] = 3
    plt.xlabel('testing data')
    plt.ylabel('Score')
    plt.scatter(range(len(x_test)), score, c=['skyblue' if y == normal else 'pink' for y in y_test])
    plt.plot([0, len(y_test)-1], [n_max,n_max],'k--', label = 'Normal Max: {0}'.format(n_max))
    plt.plot([0, len(y_test)-1], [abn_min,abn_min],'r--', label = 'Abnormal Min: {0}'.format(abn_min))
    plt.legend()
    plt.savefig('final_result/result.png')
    plt.show()
    return gan_x, normal_score, abnormal_score, score

def Evaluate_by_Category(category, g ,g_e, normal, confidence_rate = 0.99, score_rate = 0.8):  
    
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    #score = np.sum(np.abs(encoded - encoded_gan), axis = -1)
    #score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    
    
    score = Calculate_Score(score_rate, x_test, g, g_e)
    
    normal_score, abnormal_score = [], []
    for i in range(len(y_test)):
        if y_test[i]==1: normal_score.append(score[i])
        else: abnormal_score.append(score[i])
    
    normal_score, abnormal_score = np.array(normal_score), np.array(abnormal_score)
    (n_Tmin, n_Tmax) = st.t.interval(alpha=confidence_rate, df=len(normal_score)-1, loc=np.mean(normal_score), scale=st.sem(normal_score))
    (abn_Tmin, abn_Tmax) = st.t.interval(alpha=confidence_rate, df=len(abnormal_score)-1, loc=np.mean(abnormal_score), scale=st.sem(abnormal_score))
    n_max, abn_min = max(normal_score), min(abnormal_score)
    print('Interval of normal data in {0}% confidence: ({1},{2}), Max of Normal Score: {3}'
          .format(confidence_rate*100.0, n_Tmin,n_Tmax, n_max))
    print('Interval of abnormal data in {0}% confidence: ({1},{2}), Min of Abnormal Score: {3}'
          .format(confidence_rate*100.0, abn_Tmin,abn_Tmax, abn_min))
    
    rcParams['figure.figsize'] = 28, 10
    rcParams['lines.markersize'] = 3
    plt.xlabel('testing data')
    plt.ylabel('Score')
    plt.scatter(range(len(x_test)), score, c=['skyblue' if y == normal else 
                                              ('pink' if y==-1 or y==-2 or y==-3 else 
                                               ('orange' if y==-4 or y==-5 or y==-6 else
                                                ('green' if y==-7 or y==-8 or y==-9 else 'black'))) 
                                              for y in y_test])
    plt.plot([0, len(y_test)-1], [n_max,n_max],'k--', label = 'Normal Max: {0}'.format(n_max))
    plt.plot([0, len(y_test)-1], [abn_min,abn_min],'r--', label = 'Abnormal Min: {0}'.format(abn_min))
    plt.legend()
    plt.savefig('final_result/result_cateogry.png')
    plt.show()
    return gan_x, normal_score, abnormal_score, score

def plt_signal(x_data, y_data, g):
    
    x_normal = x_data[y_data==1]
    y_normal = y_data[y_data==1]
    x_Ball, x_OR, x_IR = [], [], []
    y_Ball, y_OR, y_IR = [], [], []
    for i in range(len(x_data)):
        if (y_data[i]==-1 or y_data[i]==-2 or y_data[i]==-3):
            x_Ball.append(x_data[i])
            y_Ball.append(y_data[i])
        elif (y_data[i]==-4 or y_data[i]==-5 or y_data[i]==-6):
            x_OR.append(x_data[i])
            y_OR.append(y_data[i])
        elif (y_data[i]==-7 or y_data[i]==-8 or y_data[i]==-9):
            x_IR.append(x_data[i])
            y_IR.append(y_data[i])
    x_Ball, x_OR, x_IR = np.array(x_Ball), np.array(x_OR), np.array(x_IR)
    y_Ball, y_OR, y_IR = np.array(y_Ball), np.array(y_OR), np.array(y_IR)
    
    #print(x_normal.shape, x_Ball.shape, x_OR.shape, x_IR.shape)
    #print(y_normal.shape, y_Ball.shape, y_OR.shape, y_IR.shape)
    
    n1 = random.randint(0, len(x_normal)-1)
    n2 = random.randint(0, len(x_Ball)-1)
    n3 = random.randint(0, len(x_OR)-1)
    n4 = random.randint(0, len(x_IR)-1)
    
    gan_x_normal = g.predict(x_normal)
    gan_x_Ball = g.predict(x_Ball)
    gan_x_OR = g.predict(x_OR)
    gan_x_IR = g.predict(x_IR)
    
    x_normal = x_normal.reshape(-1,1024)
    gan_x_normal = gan_x_normal.reshape(-1,1024)
    x_Ball = x_Ball.reshape(-1,1024)
    x_OR = x_Ball.reshape(-1,1024)
    x_IR = x_Ball.reshape(-1,1024)
    gan_x_Ball = gan_x_Ball.reshape(-1,1024)
    gan_x_OR = gan_x_OR.reshape(-1,1024)
    gan_x_IR = gan_x_IR.reshape(-1,1024)
    
    
    print(x_normal.shape, x_Ball.shape, x_OR.shape, x_IR.shape)
    #print(y_normal)
    #print(y_abnormal)
    
    fig, ax = plt.subplots(2,2, dpi = 150, figsize = (10, 4))
    ax[0,0].plot(x_normal[n1],'skyblue', label="X")
    ax[0,0].plot(gan_x_normal[n1],'r-', label="X~")
    ax[0,0].set_xlabel('time')
    ax[0,0].set_ylabel('Normal signal'.format(y_normal[n1]))
    ax[0,0].legend()
    
    ax[0,1].plot(x_Ball[n2],'pink', label="X")
    ax[0,1].plot(gan_x_Ball[n2],'r-', label="X~")
    ax[0,1].set_xlabel('time')
    ax[0,1].set_ylabel('Ball signal')
    ax[0,1].legend()
    
    ax[1,0].plot(x_OR[n3],'orange', label="X")
    ax[1,0].plot(gan_x_OR[n3],'r-', label="X~")
    ax[1,0].set_xlabel('time')
    ax[1,0].set_ylabel('OR signal')
    ax[1,0].legend()
    
    ax[1,1].plot(x_IR[n4],'green', label="X")
    ax[1,1].plot(gan_x_IR[n4],'r-', label="X~")
    ax[1,1].set_xlabel('time')
    ax[1,1].set_ylabel('IR signal')
    ax[1,1].legend()
    
    plt.show()

def x_to_Score(x_data, y_data, g, g_e, score_rate=0.8):
    x_data = x_data.reshape(-1,32, 32,1)
    score = Calculate_Score(score_rate, x_data, g, g_e)
    normal_score, abnormal_score = [], []
    for i in range(len(y_data)):
        if y_data[i]==1: normal_score.append(score[i])
        else: abnormal_score.append(score[i])
    normal_score, abnormal_score = np.array(normal_score), np.array(abnormal_score)
    return normal_score, abnormal_score

def SVM_evaluate(score, predict, actual):
    rcParams['figure.figsize'] = 28, 10
    rcParams['lines.markersize'] = 3
    plt.xlabel('testing data')
    plt.ylabel('Score')
    #1= normal, 0 = abnormal
    plt.scatter(range(len(score)), score, c=['red' if predict[index]!=actual[index] and actual[index] == 0
                                              else( 'blue' if predict[index]!=actual[index] and actual[index] == 1
                                                   else ('skyblue' if predict[index] == 1 
                                                         else 'pink'))
                                              for index in range(len(predict))])
    #plt.plot([0, len(y_test)-1], [n_max,n_max],'k--', label = 'Normal Max: {0}'.format(n_max))
    #plt.plot([0, len(y_test)-1], [abn_min,abn_min],'r--', label = 'Abnormal Min: {0}'.format(abn_min))
    #plt.legend()
    plt.savefig('final_result/SVM.png')
    plt.show()
    
    
def excel_generate(testX, testY, ganX):
    testX_reshape, ganX_reshape = np.reshape(testX, (len(testX), 1024)), np.reshape(ganX, (len(ganX), 1024))
    testX_label, ganX_label = np.empty((len(testX), 26), float), np.empty((len(ganX), 26), float)
    for i in range(len(testY)):
        testX_label[i] = np.append(testX_reshape[i], testY[i])
        ganX_label[i] = np.append(ganX_reshape[i], testY[i])
    
    
    df = pd.DataFrame(testX_label).T
    df.to_excel(excel_writer = 'final_result/Origin.xlsx')
    df1 = pd.DataFrame(ganX_label).T
    df1.to_excel(excel_writer = 'final_result/GAN.xlsx')
    
    
    
def generate_GIF():
    anim_file = 'ganomaly.gif'
    with imageio.get_writer(anim_file, mode='I', duration=0.5) as writer:
        filenames = glob.glob('evaluate_results/scatter*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    
    
if __name__ == "__main__":
    NORMAL, ABNORMAL = 1, -1
    GAN_TRAIN_RATE = 0.8
    SVM_TRAIN_RATE = 0.8
    SCORE_RATE = 0.8
    EPOCH = 2000
    FIG_INTERVAL = 100
    BATCH_SIZE = 16
    CONFIDENCE_RATE = 0.99
    
    loadData = LoadData()
    #(x_ok, y_ok), (x_test, y_test) = dataPreprocess_Main(train_data_ratio=0.8)
    (x_train, y_train), (x_test, y_test) = loadData.train_test_split(rate=GAN_TRAIN_RATE)
    print(x_train.shape, y_train.shape)
    print(x_test.shape,y_test.shape)
    x_train = x_train.reshape(-1,32, 32,1)
    x_test = x_test.reshape(-1,32, 32,1)
    print(x_train.shape, y_train.shape)
    print(x_test.shape,y_test.shape)
    
    ganomaly = GANomaly()
    
    #? for load trained model, use this line
    ganomaly.load_model()
    
    (g_e, g, e, f_e, d) = ganomaly.getModel()
    gantrainer = GANtrainer(g_e, g, e, f_e)
    
    gantrainer.compile()
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    d.compile(optimizer=opt, loss='binary_crossentropy')
    
    gTrainer = gantrainer.getModel()
    
    checkpoint = tf.train.Checkpoint(g_e = g_e, g = g, e = e, f_e = f_e, d = d)
    checkpoint_dir = './training_checkpoints'
    
    #! for training use this line
    #train(x_train, gTrainer, g_e, g, e, f_e, d, checkpoint, checkpoint_dir, [NORMAL,ABNORMAL], bz=BATCH_SIZE, epoch=EPOCH, fig_interval=FIG_INTERVAL)
    #ganomaly.saveModel()
    
    #generate_GIF()
    #(final_ganX, normal_score, abnormal_score, score) = final_evaluate(g, g_e, NORMAL, confidence_rate=CONFIDENCE_RATE, score_rate=SCORE_RATE)
    #excel_generate(x_test, y_test, final_ganX)
    #Evaluate_by_Category(True, g, g_e, NORMAL, confidence_rate=CONFIDENCE_RATE, score_rate=SCORE_RATE)
    plt_signal(x_data=x_test, y_data=y_test, g=g)
    
    
    
    
    
    svm_x_data, svm_y_data = loadData.SVC_dataPrepare()
    svm_normal_score, svm_abnormal_score = x_to_Score(x_data=svm_x_data, y_data=svm_y_data, g=g, g_e=g_e, score_rate=SCORE_RATE)
    print('svm normal = '+str(len(svm_normal_score))+', abnormal = '+str(len(svm_abnormal_score)))
    
    svm = SVM_classifier(normal=svm_normal_score, abnormal=svm_abnormal_score, testSize=(1-SVM_TRAIN_RATE))
    
    #! for training use this line
    #svm.train()
    
    #? for load trained model, use this line
    svm.load_model()
    
    X_test_on_SVM, predict_result, y_test_actual =  svm.accuracy_on_test()
    
    SVM_evaluate(score=X_test_on_SVM, predict=predict_result, actual=y_test_actual)
    
    svm.ConfusionMartrix()
    svm.ClassificationReport()
    