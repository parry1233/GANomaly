from unicodedata import name
from tensorflow.keras import layers
import tensorflow.keras
import tensorflow as tf

class GANomaly():
    
    def __init__(self):
        print('INIT')
        self.width = 32
        self.height = 32
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
        
    def saveModel(self):
        self.g_e.save('saved_model/G_E')
        self.g.save('saved_model/G')
        self.e.save('saved_model/E')
        self.f_e.save('saved_model/F_E')
        self.d.save('saved_model/D')
        
    def load_model(self):
        self.g_e = tf.keras.models.load_model('saved_model/G_E')
        self.g = tf.keras.models.load_model('saved_model/G')
        self.e = tf.keras.models.load_model('saved_model/E')
        self.f_e = tf.keras.models.load_model('saved_model/F_E')
        self.d = tf.keras.models.load_model('saved_model/D')
        
    #Encoder    
    def G_Encoder(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 32, 32, 1)
        
        model.add(layers.Conv2D(16,(3,3),strides=(2,2), padding='same', name='conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='leacky_1'))
        assert model.output_shape == (None, 16, 16, 16)
        
        model.add(layers.Conv2D(32,(3,3),strides=(2,2), padding='same', name='conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='norm_1'))
        model.add(layers.LeakyReLU(name='leacky_2'))
        assert model.output_shape == (None, 8, 8, 32)
        
        '''
        #! encoder 還要注意 encode 完的向量大小不要太大，不然很容易無論是無是訓練資料都可以很完美的複製回去，就沒辦法分類了。
        #! 所以在最後一層，我們過 global average pooling，讓維度快速縮小
        '''
        model.add(layers.GlobalAveragePooling2D(name='g_encoder_output'))
        assert model.output_shape == (None, 32)
        self.GEoutput = (32,)
        
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
        assert model.output_shape == (None, 32)
        model.add(layers.Dense(self.width * self.height * 2, name='dense')) #2= 32 / 4 / 4
        assert model.output_shape == (None, 2048)
        model.add(layers.Reshape((self.width//4, self.width//4, 32), name='de_reshape'))
        assert model.output_shape == (None, 8, 8, 32)
    
        model.add(layers.Conv2DTranspose(32,(3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='de_leacky_2'))
        assert model.output_shape == (None, 16, 16, 32)
        
        model.add(layers.Conv2DTranspose(64,(3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='de_leacky_3'))
        assert model.output_shape == (None, 32, 32, 64)
        
        model.add(layers.Conv2DTranspose(self.channels,(1,1), strides=(1,1), padding='same', name='decoder_deconv_output', 
                                         kernel_regularizer='l2', activation='tanh'))
        assert model.output_shape == (None, 32, 32, 1)
        
        self.Goutput = (32,32,1)
        
        print('\nGenerator Structure:')
        model.summary()
        print()
        return model
        
    def Encoder(self):
        #input_layer = layers.Input(name='input', shape=self.Goutput)
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 32, 32, 1)
       
        model.add(layers.Conv2D(16,(3,3),strides=(2,2), padding='same', name='encoder_conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='encoder_leacky_1'))
        assert model.output_shape == (None, 16, 16, 16)
        
        model.add(layers.Conv2D(32,(3,3),strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='encoder_norm_1'))
        model.add(layers.LeakyReLU(name='encoder_leacky_2'))
        assert model.output_shape == (None, 8, 8, 32)
        
        '''
        #! encoder 還要注意 encode 完的向量大小不要太大，不然很容易無論是無是訓練資料都可以很完美的複製回去，就沒辦法分類了。
        #! 所以在最後一層，我們過 global average pooling，讓維度快速縮小
        '''
        model.add(layers.GlobalAveragePooling2D(name='encoder_output'))
        assert model.output_shape == (None, 32)
        self.Eoutput = (32,)
        
        print('\nEncoder Structure:')
        model.summary()
        print()
        return model
    
    def Feature_Extractor(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        assert model.output_shape == (None, 32, 32, 1)
        
        model.add(layers.Conv2D(16, (3,3), strides=(2,2), padding='same', name='f_conv_1', kernel_regularizer='l2'))
        model.add(layers.LeakyReLU(name='f_leacky_1'))
        assert model.output_shape == (None, 16, 16, 16)
        
        model.add(layers.Conv2D(32, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization(name='f_norm_1'))
        model.add(layers.LeakyReLU(name='f_leacky_2'))
        assert model.output_shape == (None, 8, 8, 32)
        
        print('\nFeature Extractor Structure:')
        model.summary()
        print()
        return model
    
    def Discriminator(self):
        model = tf.keras.Sequential()
        model.add(self.input_layer)
        model.add(self.f_e)
        assert model.output_shape == (None, 8, 8, 32)
        
        model.add(layers.GlobalAveragePooling2D(name='glb_avg'))
        model.add(layers.Dense(1,activation='sigmoid', name='d_out'))
        assert model.output_shape == (None, 1)
        
        print('\nDiscriminator Structure:')
        model.summary()
        print()
        return model
    
    def getModel(self):
        return (self.g_e, self.g, self.e, self.f_e, self.d)
        
        