import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow as tf

class ADVloss(keras.layers.Layer): #Adversarial loss
    def __init__(self,  **kwargs):
        super(ADVloss, self).__init__(**kwargs)
    def call(self, x, mask=None):
        f_e = x[2]
        ori_feature = f_e(x[0])
        gan_feature = f_e(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

class CNTloss(keras.layers.Layer): #Content loss
    def __init__(self, **kwargs):
        super(CNTloss, self).__init__(**kwargs)
    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class ENCloss(keras.layers.Layer): #Encoder loss
    def __init__(self, **kwargs):
        super(ENCloss, self).__init__(**kwargs)
    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        g_e = x[2]
        encoder = x[3]
        return K.mean(K.square(g_e(ori) - encoder(gan)))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class APPloss(keras.layers.Layer): #Apparent loss -> Adversarial loss
    def __init__(self, **kwargs):
        super(APPloss, self).__init__(**kwargs)
    def call(self, x, mask=None):
        f_e = x[2]
        ori_feature = f_e(x[0])
        gan_feature = f_e(x[1])
        return K.mean(K.abs(ori_feature - gan_feature))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class LATloss(keras.layers.Layer): #Latent loss (without feature extractor) -> Content loss
    def __init__(self, **kwargs):
        super(LATloss, self).__init__(**kwargs)
    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
      
class GANtrainer():
    def __init__(self, g_e, g ,e, f_e):
        self.height, self.width, self.channels = 5, 5, 1
        self.g_e, self.g, self.e, self.f_e = g_e, g, e, f_e
        # model for training
        self.input_layer = layers.Input(name='input', shape=(self.height, self.width, self.channels))
        self.gan = self.g(self.input_layer) # g(x)
        
        #self.adv_loss = ADVloss(name='adv_loss')([self.input_layer, self.gan, self.f_e ])
        self.adv_loss = APPloss(name='adv_loss')([self.input_layer, self.gan, self.f_e ])
        #self.cnt_loss = CNTloss(name='cnt_loss')([self.input_layer, self.gan])
        self.cnt_loss = LATloss(name='cnt_loss')([self.input_layer, self.gan])
        #self.enc_loss = ENCloss(name='enc_loss')([self.input_layer, self.gan, self.g_e, self.e ])
        self.enc_loss = ENCloss(name='enc_loss')([self.input_layer, self.gan, self.g_e, self.g_e ])
        
        self.gan_trainer = tf.keras.Model(self.input_layer, [self.adv_loss, self.cnt_loss, self.enc_loss])
        #self.gan_trainer.summary()
    
    def loss(self, yt, yp):
        return yp
    
    def compile(self):
        losses = {
            'adv_loss': self.loss,
            'cnt_loss': self.loss,
            'enc_loss': self.loss
        }
        
        lossWeights = {'cnt_loss':50.0, 'adv_loss':1.0, 'enc_loss':1.0}
        
        #compile
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        self.gan_trainer.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)
        
        print('\nGANtrainer structure:')
        self.gan_trainer.summary()
        print()
        
    def getModel(self):
        return self.gan_trainer