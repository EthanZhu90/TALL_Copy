import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import dtypes
import tensorflow.contrib.rnn as rnn

from util.cnn import fc_layer as fc
import vs_multilayer 
from dataset_noContext import TestingDataSet
from dataset_noContext import TrainingDataSet

import pickle

class CTRL_Model(object):
    def __init__(self, batch_size, train_csv_path, test_csv_path, test_visual_feature_dir, train_visual_feature_dir,
                 word_vector_dir, useLSTM=True):
        
        self.batch_size = batch_size
        self.test_batch_size = 56
        self.vs_lr = 0.005
        self.lambda_regression = 0.01
        self.alpha = 1.0/batch_size
        self.semantic_size = 1024 # the size of visual and semantic comparison size
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 4096

        self.useLSTM = useLSTM
        self.max_words_q = 15 # check later.
        self.rnn_layer = 2
        self.lstm_input_size = 300
        self.lstm_hidden_size = 512
        self.drop_out_rate = 0.2


        # LSTM model structure
        # encoder: RNN body
        # input_size: Deprecated and unused.
        self.lstm_1 = rnn.LSTMCell(num_units=self.lstm_hidden_size, state_is_tuple=False)
        self.lstm_dropout_1 = rnn.DropoutWrapper(self.lstm_1, output_keep_prob=1 - self.drop_out_rate)
        self.lstm_2 = rnn.LSTMCell(num_units=self.lstm_hidden_size, state_is_tuple=False)
        self.lstm_dropout_2 = rnn.DropoutWrapper(self.lstm_2, output_keep_prob=1 - self.drop_out_rate)
        self.stacked_lstm = rnn.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2], state_is_tuple=False)

        # word embedding vector
        self.word2idx, self.idx2word, self.embed_ques_W = self.build_vocabulary(word_vector_dir)

        # # state-embedding
        # self.embed_state_W = tf.Variable(
        #     tf.random_uniform([2 * self.lstm_hidden_size * self.rnn_layer, self.dim_hidden], -0.08, 0.08),
        #     name='embed_state_W')
        # self.embed_state_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')

        self.train_set = TrainingDataSet(train_visual_feature_dir, train_csv_path, self.batch_size, self.word2idx, useLSTM)
        self.test_set = TestingDataSet(test_visual_feature_dir, test_csv_path, self.test_batch_size, self.word2idx, useLSTM)

    '''
    given the word vector dict, return the vocabulary
    '''
    def build_vocabulary(self, word_vector_dir):
        word_vector_dict = pickle.load(open(word_vector_dir, 'rb'))
        idx2word = list()
        word2idx = dict()
        embed = list()
        # the first word 'unk'
        word2idx['unk'] = 0
        idx2word.append('unk')
        embed.append(np.zeros(self.lstm_input_size))
        cnt = 1
        for term in word_vector_dict:
            idx2word.append(term)
            word2idx[term] = cnt
            embed.append(word_vector_dict[term])
            cnt += 1
        embed_tensor = np.vstack(embed).astype(np.float32)
        return word2idx, idx2word, embed_tensor

   
    '''
    used in training alignment model, CTRL(aln)
    '''	
    def fill_feed_dict_train(self):
        image_batch,sentence_batch,offset_batch = self.train_set.next_batch()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed
    
    '''
    used in training alignment+regression model, CTRL(reg)
    '''
    def fill_feed_dict_train_reg(self):
        image_batch, sentence_batch, offset_batch, sent_len_batch = self.train_set.next_batch_iou()
        if self.useLSTM:
            input_feed = {
                    self.visual_featmap_ph_train: image_batch,
                    self.sentence_ph_train: sentence_batch,
                    self.offset_ph: offset_batch,
                    self.sentence_ph_train_len: sent_len_batch
            }
        else:
            input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
            }
        return input_feed

    
    '''
    cross modal processing module
    '''
    def cross_modal_comb(self, visual_feat, sentence_embed, batch_size):
        vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]),
            [batch_size, batch_size, self.semantic_size])
        ss_feature = tf.reshape(tf.tile(sentence_embed,[1, batch_size]),[batch_size, batch_size, self.semantic_size])
        concat_feature = tf.reshape(tf.concat([vv_feature, ss_feature], 2),[batch_size, batch_size, self.semantic_size+self.semantic_size])
        print(concat_feature.get_shape().as_list())
        mul_feature = tf.multiply(vv_feature, ss_feature)
        add_feature = tf.add(vv_feature, ss_feature)
        
        comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature],2),[1, batch_size, batch_size, self.semantic_size*4])
        return comb_feature
    
    '''
    visual semantic inference, including visual semantic alignment and clip location regression
    '''
    def visual_semantic_infer(self, visual_feature_train, sentence_embed_train, visual_feature_test, sentence_embed_test,
                              sentence_ph_train_len, sentence_ph_test_len):

        name="CTRL_Model"
        with tf.variable_scope(name):
            print("Building training network...............................\n")
            transformed_clip_train = fc('v2s_lt', visual_feature_train, output_dim=self.semantic_size) 
            transformed_clip_train_norm = tf.nn.l2_normalize(transformed_clip_train, dim=1)

            if self.useLSTM:
                sentence_embed_train = self.lstm_embed(sentence_embed_train, sentence_ph_train_len)

            transformed_sentence_train = fc('s2s_lt', sentence_embed_train, output_dim=self.semantic_size)
            transformed_sentence_train_norm = tf.nn.l2_normalize(transformed_sentence_train, dim=1)  
            cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm, transformed_sentence_train_norm, self.batch_size)
            sim_score_mat_train = vs_multilayer.vs_multilayer(cross_modal_vec_train, "vs_multilayer_lt", middle_layer_dim=1000)
            sim_score_mat_train = tf.reshape(sim_score_mat_train,[self.batch_size, self.batch_size, 3])

            tf.get_variable_scope().reuse_variables()
            print("Building test network...............................\n")
            transformed_clip_test = fc('v2s_lt', visual_feature_test, output_dim=self.semantic_size)
            transformed_clip_test_norm = tf.nn.l2_normalize(transformed_clip_test, dim=1)

            if self.useLSTM:
                sentence_embed_test = self.lstm_embed(sentence_embed_test, sentence_ph_test_len)
            transformed_sentence_test = fc('s2s_lt', sentence_embed_test, output_dim=self.semantic_size)
            transformed_sentence_test_norm = tf.nn.l2_normalize(transformed_sentence_test, dim=1)

            cross_modal_vec_test = self.cross_modal_comb(transformed_clip_test_norm, transformed_sentence_test_norm, self.test_batch_size)
            sim_score_mat_test = vs_multilayer.vs_multilayer(cross_modal_vec_test, "vs_multilayer_lt", reuse=True, middle_layer_dim=1000)
            sim_score_mat_test = tf.reshape(sim_score_mat_test, [self.test_batch_size, self.test_batch_size, 3])

            cross_modal_vec_test_1 = self.cross_modal_comb(tf.reshape(transformed_clip_test_norm[1], shape=(1,1024)),
                                                           tf.reshape(transformed_sentence_test_norm[1], shape=(1,1024)), 1)
            sim_score_mat_test_1 = vs_multilayer.vs_multilayer(cross_modal_vec_test_1, "vs_multilayer_lt", reuse=True, middle_layer_dim=1000)
            sim_score_mat_test_1 = tf.reshape(sim_score_mat_test_1, [3])
            return sim_score_mat_train, sim_score_mat_test, sim_score_mat_test_1

    def lstm_embed(self, sentences, sentence_ph_train_len):

        # state = [tf.zeros([self.batch_size, x]) for x in [self.lstm_hidden_size, self.lstm_hidden_size]]
        sent_1dim = tf.reshape(sentences, [-1, 1])
        sent_vector_2dim = tf.gather_nd(self.embed_ques_W, sent_1dim)
        sent_vector = tf.reshape(sent_vector_2dim, [int(sentences.shape[0]), int(sentences.shape[1]), -1])
        # embedding_lookup must contain a variable.
        # sent_vector = tf.nn.embedding_lookup(self.embed_ques_W, [int(sentences.shape[0]), int(sentences.shape[1]), -1])
        state = self.stacked_lstm.zero_state(sentences.shape[0], tf.float32)
        # inputs:[batch_size, max_time, size] if time_major = Flase.
        output, state = tf.nn.dynamic_rnn(self.stacked_lstm, inputs=sent_vector, sequence_length=sentence_ph_train_len,
                                          initial_state=state, dtype=tf.float32, time_major=False)

        state_drop = tf.nn.dropout(state, 1 - self.drop_out_rate)
        # state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        # state_emb = tf.tanh(state_linear)

        return state_drop


    '''
    compute alignment and regression loss
    '''
    def compute_loss_reg(self, sim_reg_mat, offset_label):

        sim_score_mat, p_reg_mat, l_reg_mat = tf.split(sim_reg_mat, num_or_size_splits=3, axis=2)
        sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size])
        l_reg_mat = tf.reshape(l_reg_mat, [self.batch_size, self.batch_size])
        p_reg_mat = tf.reshape(p_reg_mat, [self.batch_size, self.batch_size])
        # unit matrix with -2
        I_2 = tf.diag(tf.constant(-2.0, shape=[self.batch_size]))
        all1 = tf.constant(1.0, shape=[self.batch_size, self.batch_size])
        #               | -1  1   1...   |

        #   mask_mat =  | 1  -1  -1...   |

        #               | 1   1  -1 ...  |
        mask_mat = tf.add(I_2, all1)
        # loss cls, not considering iou
        I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
        batch_para_mat = tf.constant(self.alpha, shape=[self.batch_size, self.batch_size])

        para_mat = tf.add(I,batch_para_mat)
        loss_mat = tf.log(tf.add(all1, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
        loss_mat = tf.multiply(loss_mat, para_mat)
        loss_align = tf.reduce_mean(loss_mat)
        # regression loss

        l_reg_diag = tf.matmul(tf.multiply(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        p_reg_diag = tf.matmul(tf.multiply(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        offset_pred = tf.concat((p_reg_diag, l_reg_diag), 1)
        loss_reg = tf.reduce_mean(tf.abs(tf.subtract(offset_pred, offset_label)))

        loss=tf.add(tf.multiply(self.lambda_regression, loss_reg), loss_align)
        return loss, offset_pred, loss_reg


    def init_placeholder(self):
        visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim))
        if self.useLSTM:
            # using LSTM, input is the idx of word
            sentence_ph_train = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_words_q))
            sentence_ph_train_len = tf.placeholder(tf.int32, shape=(self.batch_size))
        else:
            sentence_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.sentence_embedding_size))
            sentence_ph_train_len = -1

        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size,2))
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim))

        if self.useLSTM:
            # using LSTM, input is the idx of word
            sentence_ph_test = tf.placeholder(tf.int32, shape=(self.test_batch_size, self.max_words_q))
            sentence_ph_test_len = tf.placeholder(tf.int32, shape=(self.test_batch_size))
        else:
            sentence_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))
            sentence_ph_test_len = -1
        return visual_featmap_ph_train,sentence_ph_train,offset_ph,visual_featmap_ph_test, sentence_ph_test, \
               sentence_ph_train_len, sentence_ph_test_len
    

    def get_variables_by_name(self,name_list):
        v_list = tf.trainable_variables()
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print("Variables of <"+name+">")
            for v in v_dict[name]:
                print("    "+v.name)
        return v_dict

    def training(self, loss):
        
        v_dict = self.get_variables_by_name(["lt"])
        vs_optimizer = tf.train.AdamOptimizer(self.vs_lr, name='vs_adam')
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["lt"])
        return vs_train_op


    def construct_model(self):


        # initialize the placeholder
        self.visual_featmap_ph_train, self.sentence_ph_train, self.offset_ph, self.visual_featmap_ph_test, self.sentence_ph_test, \
        self.sentence_ph_train_len, self.sentence_ph_test_len =self.init_placeholder()
        # build inference network
        sim_reg_mat, sim_reg_mat_test, sim_reg_mat_test_1 = self.visual_semantic_infer(self.visual_featmap_ph_train, self.sentence_ph_train,
                                                                   self.visual_featmap_ph_test, self.sentence_ph_test,
                                                                   self.sentence_ph_train_len, self.sentence_ph_test_len)
        # compute loss
        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_reg_mat, self.offset_ph)
        # optimize
        self.vs_train_op = self.training(self.loss_align_reg)
        return self.loss_align_reg, self.vs_train_op, sim_reg_mat_test, sim_reg_mat_test_1,  offset_pred, loss_reg


