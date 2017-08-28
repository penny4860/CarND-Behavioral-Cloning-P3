
import pickle
import matplotlib.pyplot as plt

def plot_history(pkl_file):
    history = pickle.load(open(pkl_file, 'rb'))
    ### plot the training and validation loss for each epoch
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


plot_history('training_history.pkl')
