from unicodedata import name
from tensorflow.keras import layers
import tensorflow.keras
import tensorflow as tf

class GANomaly():
    
    def __init__(self):
        print('INIT')
        self.width = 5
        self.height = 5
        self.channels = 1
        self.input_layer = layers.Input(name='input', shape=(self.height,self.width,self.channels))
        self.GEoutput = (0,0,0)
        self.Goutput = (0,0,0)
        self.Eoutput = (0,0,0)
        self.g_e = self.G_Encoder()
        self.g = self.Generator()
        self.e = self.Encoder()
        self.f_e = self.Feature_Extractor()
        self.d = self.Discriminator()
        
    def load(self,g_e, g, e, f_e, d):
        self.g_e = g_e
        self.g = g
        self.e = e
        self.f_e = f_e
        self.d = d
        
    #Encoder    
    def G_Encoder(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 5, 5, 1)
        
        model.add(layers.Conv2D(5,(2,2),strides=(1,1), padding='same', name='conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='leacky_1'))
        assert model.output_shape == (None, 5, 5, 5)
        
        model.add(layers.Conv2D(10,(2,2),strides=(1,1), padding='same', name='conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='norm_1'))
        model.add(layers.LeakyReLU(name='leacky_2'))
        assert model.output_shape == (None, 5, 5, 10)
        
        model.add(layers.Conv2D(20,(2,2),strides=(1,1), padding='same', name='conv_3', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='norm_2'))
        model.add(layers.LeakyReLU(name='leacky_3'))
        assert model.output_shape == (None, 5, 5, 20)
        
        '''
        #! encoder 還要注意 encode 完的向量大小不要太大，不然很容易無論是無是訓練資料都可以很完美的複製回去，就沒辦法分類了。
        #! 所以在最後一層，我們過 global average pooling，讓維度快速縮小
        '''
        model.add(layers.GlobalAveragePooling2D(name='g_encoder_output'))
        assert model.output_shape == (None, 20)
        self.GEoutput = (20,)
        
        print('\nG_Encoder Structure:')
        model.summary()
        print('')
        return model
    
    #Generator
    def Generator(self):
        #input_layer = layers.Input(name='input', shape=self.GEoutput) #input shape must match output shape of G_Encoder()
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        model.add(self.g_e)
        assert model.output_shape == (None, 20)
        model.add(layers.Dense(self.width * self.height * 5, name='dense')) #5= 20 / 2 / 2
        assert model.output_shape == (None, 125)
        model.add(layers.Reshape((self.width//1, self.width//1, 5), name='de_reshape'))
        assert model.output_shape == (None, 5, 5, 5)
        
        model.add(layers.Conv2DTranspose(20,(2,2), strides=(1,1), padding='same', name='deconv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='de_leacky_1'))
        assert model.output_shape == (None, 5, 5, 20)
    
        model.add(layers.Conv2DTranspose(10,(2,2), strides=(1,1), padding='same', name='deconv_2', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='de_leacky_2'))
        assert model.output_shape == (None, 5, 5, 10)
        
        model.add(layers.Conv2DTranspose(5,(2,2), strides=(1,1), padding='same', name='deconv_3', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='de_leacky_3'))
        assert model.output_shape == (None, 5, 5, 5)
        
        model.add(layers.Conv2DTranspose(self.channels,(1,1), strides=(1,1), padding='same', name='decoder_deconv_output', 
                                         kernel_regularizer='l2', activation='tanh'))
        assert model.output_shape == (None, 5, 5, 1)
        
        self.Goutput = (5,5,1)
        
        print('\nGenerator Structure:')
        model.summary()
        print()
        return model
        
    def Encoder(self):
        #input_layer = layers.Input(name='input', shape=self.Goutput)
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 5, 5, 1)
       
        model.add(layers.Conv2D(5,(2,2),strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='encoder_leacky_1'))
        assert model.output_shape == (None, 5, 5, 5)
        
        model.add(layers.Conv2D(10,(2,2),strides=(1,1), padding='same', name='encoder_conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='encoder_norm_1'))
        model.add(layers.LeakyReLU(name='encoder_leacky_2'))
        assert model.output_shape == (None, 5, 5, 10)
        
        model.add(layers.Conv2D(20,(2,2),strides=(1,1), padding='same', name='encoder_conv_3', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='encoder_norm_2'))
        model.add(layers.LeakyReLU(name='encoder_leacky_3'))
        assert model.output_shape == (None, 5, 5, 20)
        
        '''
        #! encoder 還要注意 encode 完的向量大小不要太大，不然很容易無論是無是訓練資料都可以很完美的複製回去，就沒辦法分類了。
        #! 所以在最後一層，我們過 global average pooling，讓維度快速縮小
        '''
        model.add(layers.GlobalAveragePooling2D(name='encoder_output'))
        assert model.output_shape == (None, 20)
        self.Eoutput = (20,)
        
        print('\nEncoder Structure:')
        model.summary()
        print()
        return model
    
    def Feature_Extractor(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 5, 5, 1)
        
        model.add(layers.Conv2D(5, (2,2), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='f_leacky_1'))
        assert model.output_shape == (None, 5, 5, 5)
        
        model.add(layers.Conv2D(10, (2,2), strides=(1,1), padding='same', name='f_conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='f_norm_1'))
        model.add(layers.LeakyReLU(name='f_leacky_2'))
        assert model.output_shape == (None, 5, 5, 10)
        
        model.add(layers.Conv2D(20, (2,2), strides=(1,1), padding='same', name='f_conv_3', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='f_norm_2'))
        model.add(layers.LeakyReLU(name='f_leacky_3'))
        assert model.output_shape == (None, 5, 5, 20)
        
        print('\nFeature Extractor Structure:')
        model.summary()
        print()
        return model
    
    def Discriminator(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        model.add(self.f_e)
        assert model.output_shape == (None, 5, 5, 20)
        
        model.add(layers.GlobalAveragePooling2D(name='glb_avg'))
        model.add(layers.Dense(1,activation='sigmoid', name='d_out'))
        assert model.output_shape == (None, 1)
        
        print('\nDiscriminator Structure:')
        model.summary()
        print()
        return model
    
    def getModel(self):
        return (self.g_e, self.g, self.e, self.f_e, self.d)
        
        