import tensorflow as tf
from Utility.Global_definition import *
from Utility.Model_Parameter import *
from CNN_MODEL.CNN_qa import *
from Utility.DataTools import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('margin', 'Model/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/QA',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('model_dir', 'Model/',
                           """Directory where to write event logs """
                           """and checkpoint.""")



def train(if_train):

        vocab = build_vocab()  #dictionary
        alist = read_alist() #Answer List
        raw = read_raw()     # raw list
        #shape = [batch size]
        train_q_x = tf.placeholder(tf.int32, shape=[None,QUESTION_LEN])
        train_a_x  = tf.placeholder(tf.int32,shape=[None,QUESTION_LEN])
        train_n_x  = tf.placeholder(tf.int32,shape=[None,QUESTION_LEN])



        q,a,ne = inference(train_q_x,train_a_x,train_n_x,vocab_size=len(vocab),embedding_size=EMBEDDING_SIZE,questionlen=QUESTION_LEN) #m= margin,q= question,a = answer,ne= negative answer

        loss_op = loss(q,a,ne)

        train_op = train_in_cnn(loss_op)

        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)

        tf.scalar_summary("loss", loss_op)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')

        for step in range(FLAGS.max_steps):
            q,a,ne= next_batch(vocab=vocab,alist=alist,raw=raw,size=BATCH_SIZE)
            _,loss_value = sess.run([train_op,loss_op], feed_dict={train_q_x:q, train_a_x:a,train_n_x:ne})
            train_writer.add_summary(sess.run(merge, feed_dict={train_q_x:q, train_a_x:a,train_n_x:ne}), step)


def main(argv=None):
    train(if_train=True)




if __name__ == '__main__':
  tf.app.run()