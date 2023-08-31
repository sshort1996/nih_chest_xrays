# import evertyhing from MNIST dir
import sys
sys.path.append('/Users/ShaneShort/Documents/nih_chest_xrays')

from MNIST import preProcessor, compile, test

# run preprocess to set up data 
X_test, X_train, X_val, Y_test, Y_train, Y_val = preProcessor.pre_process()

# configure model 
model = compile.compile_model()

# train model 
history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=15,
                    batch_size=512)

# analyse training data and test model
test_loss, test_acc = test.analyse_history(model, history, X_test, Y_test)