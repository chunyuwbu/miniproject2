from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "C:/Users/14455/Desktop/flower_photos1/vegetable/282691427_8402fc8c88.jpg"
path2 = "C:/Users/14455/Desktop/flower_photos1/vegetable/136470621_8ed6dea155.jpg"
path3 = "C:/Users/14455/Desktop/flower_photos1/roses/394990940_7af082cf8d_n.jpg"
path4 = "C:/Users/14455/Desktop/flower_photos1/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "C:/Users/14455/Desktop/flower_photos1/sunflowers/2950505226_529e013bf7_m.jpg"

flower_dict = {0:'roses',1:'sunflowers',2:'vegetable'}

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('C:/Users/14455/Desktop/models/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('C:/Users/14455/Desktop/models'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    # Print the prediction matrix
    print(classification_result)
    # Print an index of the maximum value of each row of the prediction matrix
    print(tf.argmax(classification_result,1).eval())
    # Classification of items by dictionary according to index
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"个东西预测:"+flower_dict[output[i]])