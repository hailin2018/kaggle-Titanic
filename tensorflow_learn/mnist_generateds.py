#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import os


image_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'							#train图像路径 NOTE:此时的图像路径只到文件夹,没有具体到图像名
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'						#train标签路径
tfRecord_train='./data/mnist_train.tfrecords'										#train转换得到的二进制文件的路径

image_test_path='./mnist_data_jpg/mnist_test_jpg_10000/'							#同上
label_test_path='./mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test='./data/mnist_test.tfrecords'

data_path='./data'																	#??转换得到的tfrecord文件的保存路径???NOTE

resize_height = 28
resize_width = 28



def write_tfRecord(tfRecordName, image_path, label_path):							#定义函数:生成二进制数据,三个参数:tfrecord文件名, train图像文件的路径, train图像标签文件的路径
    writer = tf.python_io.TFRecordWriter(tfRecordName)  							#用函数1新建一个writer,函数接收一个参数: tfrecord文件名
    num_pic = 0 																	#定义一个计数器,目的:为了显示进度

    f = open(label_path, 'r')														#open()接收一个参数:标签文件路径,以读的方式打开标签文件
    contents = f.readlines()														#标签文件是一个TXT文件,以行存放标签,每行由图片名\标签组成,中间空格隔开
																					#所以用readlines()读取标签文件,以列表形式返回
    f.close()																		#关闭标签文件

    for content in contents:														#for循环遍历标签文件
        value = content.split()														#split()剥离图片名\标签
        img_path = image_path + value[0] 											#用图片路径+图片名组合成具体的图片路径
        img = Image.open(img_path)													#用刚刚得到的具体的图片路径和Image.open()打开图片,
        img_raw = img.tobytes() #NOTE 1												#tobytes()把图片转换为二进制

        labels = [0] * 10  															#因为标签文件中的标签value[1]是一个数,需要转为一个10维的向量 
        labels[int(value[1])] = 1 #NOTE 2											#先定义一个10维的0向量,再把对应的元素赋值为1,
            

        example = tf.train.Example(features=tf.train.Features(feature={				# tf.train.Example()作用:以键值对的形式存储数据,即把图片和标签封装到example中
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),				 					#接收参数:NOTE 1 img_raw NOTE 2 labels
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 

        writer.write(example.SerializeToString())									#SerializeToString()把example序列化成字符串存储

        num_pic += 1 																#计数器,每转换一张图片,计数器+1,并打印进度
        print ("the number of picture:", num_pic)

    writer.close()																	#for循环转换完所有的图片后,关闭writer
    print("write tfrecord successful")



def generate_tfRecord():																#定义函数,功能是:把图片和标签转换为二进制文件
	isExists = os.path.exists(data_path) 												#os模块: 判断保存路径是否讯在?NOTE  是谁的路径??
	if not isExists: 
 		os.makedirs(data_path)															#如果不存在,用os模块函数新建路径
		print 'The directory was created successfully'
	else:
		print 'directory already exists' 
	write_tfRecord(tfRecord_train, image_train_path, label_train_path)					#调用write_tfRecord(),把train数据转换为二进制文件
 	write_tfRecord(tfRecord_test, image_test_path, label_test_path)						#三个参数: tfrecord文件名, train图像的路径, train图像标签的路径
  


#上面的两个函数作用:把train和test的图片和标签转换为tfrecord文件      write_tfRecord()是利用for循环,以单个图片为单位转换为tfrecord文件
#######################################################################################################################################################################################################
#下面的函数作用: 批量读取tfrecord文件								 read_tfRecord()是一次读取read_tfRecord文件中所有的图片\标签



def read_tfRecord(tfRecord_path):														#定义函数,实现具体的读取,一个参数:待读取文件的保存路径
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)		#1.tf.train.string_input_producer(),新建文件队列名??  接收两个参数:列表存放读取路径,
    reader = tf.TFRecordReader()														#新建reader
    _, serialized_example = reader.read(filename_queue) 								#2.reader.read(),接收文件队列名,读取文件中的所有样本并保存在serialized_example
    features = tf.parse_single_example(serialized_example,								#3.tf.parse_single_example(),接收serialized_example,进行解序列化,键名应该和保存时的一样
                                       features={
                                        'label': tf.FixedLenFeature([10], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })

    img = tf.decode_raw(features['img_raw'], tf.uint8)									#图片和标签在tfrecord文件中被序列化成字符串存储,所以读取出来后要还原
    img.set_shape([784])																#tf.decode_raw()把字符串类型的图片转为8位无符号整型,再改变shape为1行784列
    img = tf.cast(img, tf.float32) * (1. / 255)											#把图片的所有像素值变为[0,1]浮点型

    label = tf.cast(features['label'], tf.float32)										#把标签列表变为浮点数
    return img, label 																	#返回读取出的图片和标签
      


def get_tfrecord(num, isTrain=True):													#定义函数,批量读取tfrecord文件,有两个参数:每次读取的数量,是否是train数据,默认是True
    if isTrain:																			#根据第二个变量的值,给待读取tfrecord文件的路径变量tfRecord_path赋值
        tfRecord_path = tfRecord_train													#如果是train数据,则赋值train data的tfrecord文件路径
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)											#调用read_tfRecord(),实现tfrecord文件的读取 NOTE 一次读取所有的图片和标签

    img_batch, label_batch = tf.train.shuffle_batch([img, label],						#tf.train.shuffle_batch()从[img, label]中随机读取一组batch的数据
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)
    return img_batch, label_batch														#返回一组batch的图片和标签


def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()










