import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import mnist_generateds#NOTE 2 导入数据集生成文件

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"

train_num_examples = 60000#NOTE 3 样本的数量要手动设定,之前直接从mnist数据集中获取数据时,可调用mnist.train.num_examples得到样本总数
						  #       用于计算指数衰减学习率的更新频率=样本数/batch size

def backward():

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False) 

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)	#NOTE 4 调用mnist_generateds.get_tfrecord()获取一组batch数据

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)							#NOTE 1 断点续训
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
			
        coord = tf.train.Coordinator()													#NOTE 5	利用多线程提高图片和标签的批获取效率
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)					#	   	开启线程协调器	
        
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])									#NOTE 4	在会话中run()获取一组batch数据,之前直接从mnist数据集中获取数据时,
																						#		可直接调用mnist.train.next_batch(BATCH_SIZE)从mnist数据中随机获取一组batch数据
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()															#NOTE 5 利用多线程提高图片和标签的批获取效率
        coord.join(threads)																#		关闭线程协调器


def main():
    backward()#9

if __name__ == '__main__':
    main()


