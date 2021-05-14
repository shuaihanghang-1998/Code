import os
import tensorflow.compat.v1 as tf 
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
from scipy import signal
from CNN import CNN
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.app.flags.DEFINE_integer("epoch", 10, "epoch of training step")
tf.app.flags.DEFINE_integer("batch_size", 128, "mini_batch_size")

def get_batches(X, y):
    batch_size = FLAGS.batch_size
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if (i+batch_size) < len(X) else len(X)
        yield X[begin_i:end_i], y[begin_i:end_i]

def get_file():
    file_dir = "C:/Users/86528/Desktop/gaitRecognition-master/data"
    file_dir2 = "C:/Users/86528/Desktop/gaitRecognition-master/data2"
    X = []
    Y = []
    acc = os.listdir(file_dir)
    gyr = os.listdir(file_dir2)
    for i in range(10):
        f = open(file_dir + '/' + acc[i])
        f2 = open(file_dir2 + '/' + gyr[i])
        line = f.readlines()
        line2 = f2.readlines()
        temp = []
        line_wrong = 0
        for num in range(len(line)):
            if num < 150:
                time, x, y, z = [float(i) for i in line[25].split()]
                time2, x2, y2, z2 = [float(i) for i in line2[num].split()]
                if time-time2 < 30 and time-time2 > -30:
                    line_wrong=25-num;
                continue
            time, x, y, z = [float(i) for i in line[num].split()]
            time2, x2, y2, z2 = [float(i) for i in line2[num - line_wrong].split()]
            temp.append([x, y, z, x2, y2, z2])
            num += 1
        b, a = signal.butter(10, 0.1, 'lowpass')
        temp = signal.filtfilt(b, a, temp, axis=0)
        group = []
        for x in temp:
            group.append(x)
            if len(group) == 50:
                X.append(group)
                Y.append(i)
                group = group[25:]
    return X, Y

if __name__ == "__main__":
    log_dir = "D:/Code/ml practical training/gaitRecognition-master/log/"
    X, Y = get_file()
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    print(X.shape)
    print(Y.shape)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=40)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=40)
    print(train_x.shape)
    print(test_x.shape)
    print(valid_x.shape)
    model = CNN()
    model.build_net()
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=log_dir, is_chief=True, saver=saver, summary_op=None,
                             save_summaries_secs=None,save_model_secs=None, global_step=model.global_step)
    sess_context_manager = sv.prepare_or_wait_for_session()
    maxAcc = 0
    with sess_context_manager as sess:
        print("----------------train----------------")
        print(time.strftime('%Y-%m-%d %H:%M:%S'))
        # summary_writer = tf.summary.FileWriter(log_dir)
        for e in range(FLAGS.epoch):
            train_x, train_y = shuffle(train_x, train_y)
            for xs, ys in get_batches(train_x, train_y):
                feed_dict = {model.x: xs, model.y_: ys}
                _, loss, step, train_acc = sess.run(
                    [model.train_op, model.loss, model.global_step, model.acc], feed_dict=feed_dict)
                if step % 10 == 0:
                    feed_dict = {model.x: valid_x, model.y_: valid_y}
                    test_acc = sess.run(model.acc, feed_dict=feed_dict)
                    print("epoch->{:<3} step->{:<5} loss:{:<10.5} train_acc:{:<10.2%} "
                            "valid_acc:{:<10.2%} maxAcc:{:<10.2%}".
                            format(e, step, loss, train_acc, test_acc, maxAcc))
                    # summary_writer.add_summary(merged_summary, step)
                    if test_acc > maxAcc:
                        maxAcc = test_acc
                        saver.save(sess=sess, save_path=log_dir, global_step=step)
        print("---------------training has finished!----------------")
        print(time.strftime('%Y-%m-%d %H:%M:%S'))
        print("-------------------predict---------------")
        model_file = tf.train.latest_checkpoint(log_dir)
        saver.restore(sess, model_file)
        feed_dict = {model.x: test_x, model.y_: test_y}
        acc = sess.run(model.acc, feed_dict=feed_dict)
        print("test_acc:{:.2%}".format(acc))
