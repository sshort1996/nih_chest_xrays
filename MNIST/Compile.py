from preprocess import import_data, prep_data, plot_data
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k


def preprocess():
    x_train, y_train, x_test, y_test, inpx, img_cols, img_rows = import_data()
    x_train, y_train, x_test, y_test = prep_data(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test, inpx, img_cols, img_rows

def compile(inpx):
    
    inpx = Input(shape=inpx)
    layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
    layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
    layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
    layer4 = Dropout(0.5)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(250, activation='sigmoid')(layer5)
    layer7 = Dense(10, activation='softmax')(layer6)
    return layer7

if __name__=="__main__":
    x_train, y_train, x_test, y_test, inpx, img_cols, img_rows = preprocess()
    layer7 = compile(inpx)
    # model = train(inpx, layer7, x_train, y_train)
    # score = test(model, x_test, y_test)