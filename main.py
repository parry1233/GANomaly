from sqlalchemy import null
from GANomaly import GANomaly
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
    
def train(x_ok, gan_trainer, d, g, g_e, cp, cpdir, classType, bz=32, epoch=1000):
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
        
        if (i+1) % 10 == 0:
            print(f'epoch: {i+1}, g_loss: {g_loss}, d_loss: {d_loss}')
            save_checkpoints(cp, cpdir)
            evaluate_fig(g, g_e, i+1, classType)
            
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
            
def evaluate_fig(g, g_e, epoch, classType):
    normal, abnormal = classType[0], classType[1]
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    score = np.sum(np.abs(encoded - encoded_gan), axis = -1)
    score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    rcParams['figure.figsize'] = 14, 5
    plt.scatter(range(len(x_test)), score, c=['blue' if x == normal else 'red' for x in y_test])
    plt.title('epoch: {:04d}'.format(epoch))
    #plt.show()
    # save plot to file
    filename = 'evaluate_results/scatter_plot_{:04d}.png'.format(epoch)
    plt.savefig(filename)
    plt.close()
    
def final_evaluate(g,g_e, normal):
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    score = np.sum(np.abs(encoded - encoded_gan), axis = -1)
    score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
    rcParams['figure.figsize'] = 14, 5
    plt.scatter(range(len(x_test)), score, c=['blue' if x == normal else 'red' for x in y_test])
    plt.show()
    return gan_x
    
def pic(ganX):
    i = 5 # or 1
    image = np.reshape(ganX[i:i+1], (64, 64))
    image = image * 127 + 127
    plt.imshow(image.astype(np.uint8), cmap='gray')
    plt.show()
    
    
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
    normal, abnormal = 1, -1
    (x_ok, y_ok), (x_test, y_test) = dataPreprocess_Main(train_data_ratio=0.8)
    #x_ok = loadData.reshape_x(x_ok, 64, 64)
    #x_test = loadData.reshape_x(x_test, 64, 64)
    
    ganomaly = GANomaly()
    (g_e, g, e, f_e, d) = ganomaly.getModel()
    gantrainer = GANtrainer(g_e, g, e, f_e)
    
    gantrainer.compile()
    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
    d.compile(optimizer=opt, loss='binary_crossentropy')
    
    gTrainer = gantrainer.getModel()
    
    checkpoint = tf.train.Checkpoint(g_e = g_e, g = g, e = e, f_e = f_e, d = d)
    checkpoint_dir = './training_checkpoints'
    
    train(x_ok, gTrainer, d, g, g_e ,checkpoint, checkpoint_dir, [normal,abnormal], bz=16, epoch=500)
    generate_GIF()
    final_ganX = final_evaluate(g,g_e, normal)
    #pic(final_ganX)