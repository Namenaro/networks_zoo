# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np

#plt.xkcd()
def get_ecg_test_sample(num_patient):
    sample = x_test[num_patient,:,:,:]
    print("форма тензора с экг: "+ str(sample.shape))
    return sample

def reshape_ecg_tensor(ecg):
    # превратим (252, 1, 12) в (12, 252)
    print ("форма тезора с экг =  " + str(ecg.shape))
    ecg = ecg[:,0,:]
    ecg = np.transpose(ecg)
    print ("форма тезора с экг (после напильника) =" + str(ecg))
    return ecg

def draw_reconstruction_to_png(ecg_true, ecg_predicted, png_filename):
    """

    :param ecg_true: истинная экг
    :param ecg_predicted: предсказанное
    :param png_filename: имя для файла с картинкой
    :return:
    """
    ecg_true = reshape_ecg_tensor(ecg_true)
    ecg_predicted = reshape_ecg_tensor(ecg_predicted)

    assert ecg_true.shape == ecg_predicted.shape

    len_of_time = len(ecg_true[0])
    t = [i for i in range(len_of_time)]


    rows = len(ecg_true)  # кол-во каналов
    cols = 2              # true и predicted - 2 столбика
    f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    for i in range(rows):
        true_i_chanel = ecg_true[i]
        predicted_i_chanel = ecg_predicted[i]
        axarr[i, 0].plot(t, true_i_chanel)
        axarr[i, 1].plot(t, predicted_i_chanel)


    plt.savefig(png_filename+".png")



def show_reconstruction_by_ae(ecg_sample, name):
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)

    ecg_sample = np.array([ecg_sample])
    prediction = trained_model.predict(ecg_sample)

    draw_reconstruction_to_png(ecg_sample[0],prediction[0], name)


    
def save_history(history,caterpillar_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #caterpillar.save('ctp.h5')
    #caterpillar.save_weights('ctp_weights.h5')
    plt.savefig(caterpillar_name+".png")
    

def gpu_cpu_test():
	#---GPU/CPU test and rate status-----
#Also useful to see computation rate of device
	import tensorflow as tf
	print ("TF version - "+ tf.__version__)
	if tf.test.gpu_device_name():
		print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
		print("---------- -----------")
	else:
		print("IMPORTANT: Computing without GPU here.")
		print("---------- -----------")

	from tensorflow.python.client import device_lib
	print("Using device info: ")
	print(device_lib.list_local_devices())
