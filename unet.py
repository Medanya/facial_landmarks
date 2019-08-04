import keras
from keras.layers import Conv2D, MaxPooling2D,Flatten, Dropout, Input, BatchNormalization, Activation, Conv2DTranspose, concatenate
from keras.models import Model
from keras.optimizers import Adam


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def unet_small(input_shape, n_filters=16, dropout=0.5, batchnorm=True, lr=0.001, out_channels=2, loss="mean_absolute_error"):
    # contracting path
    input_img = Input(input_shape, name='img')
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c4)
    u6 = concatenate([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(out_channels, (3,3), activation='sigmoid', padding="same") (c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    optimizer = Adam(lr=lr)
    model.compile(loss=loss,optimizer=optimizer)
    return model
