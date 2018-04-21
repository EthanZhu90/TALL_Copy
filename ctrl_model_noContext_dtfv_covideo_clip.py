import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from util.cnn import fc_layer as fc
import vs_multilayer 
from dataset_noContext_dtfv_covideo_clip import TestingDataSet
from dataset_noContext_dtfv_covideo_clip import TrainingDataSet


class CTRL_Model(object):
    def __init__(self, batch_size, train_csv_path, test_csv_path, test_visual_feature_dir, train_visual_feature_dir):
        
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.vs_lr = 0.005
        self.lambda_regression = 0.01
        self.alpha = 1.0/batch_size
        self.semantic_size = 2048  # 3072  # 2048  # 2048 # the size of visual and semantic comparison size
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 21800  # 43600
        self.train_set=TrainingDataSet(train_visual_feature_dir, train_csv_path, self.batch_size)
        self.test_set=TestingDataSet(test_visual_feature_dir, test_csv_path, self.test_batch_size)
   
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
        image_batch_pos, image_batch_neg, sentence_batch, offset_batch = self.train_set.next_batch_iou()
        input_feed = {
                self.visual_featmap_ph_train_pos: image_batch_pos,
                self.visual_featmap_ph_train_neg: image_batch_neg,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed

    
    '''
    cross modal processing module
    '''
    def cross_modal_comb_toremve(self, visual_feat, sentence_embed, batch_size):
        vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]),
            [batch_size, batch_size, self.semantic_size])
        ss_feature = tf.reshape(tf.tile(sentence_embed,[1, batch_size]),[batch_size, batch_size, self.semantic_size])
        concat_feature = tf.reshape(tf.concat([vv_feature, ss_feature], 2),[batch_size, batch_size, self.semantic_size+self.semantic_size])
        print(concat_feature.get_shape().as_list())
        mul_feature = tf.multiply(vv_feature, ss_feature)
        add_feature = tf.add(vv_feature, ss_feature)
        
        comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature],2),[1, batch_size, batch_size, self.semantic_size*4])
        return comb_feature



    def cross_modal_comb(self, visual_feat, sentence_embed, batch_size):
        concat_feature = tf.concat([visual_feat, sentence_embed], 1)
        # print(concat_feature.get_shape().as_list())
        mul_feature = tf.multiply(visual_feat, sentence_embed)
        add_feature = tf.add(visual_feat, sentence_embed)
        comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature], 1),
                                  [1, 1, visual_feat.get_shape().as_list()[0], self.semantic_size*4])

        return comb_feature
    '''
    visual semantic inference, including visual semantic alignment and clip location regression
    '''
    def visual_semantic_infer(self, visual_feature_train_pos, visual_feature_train_neg, sentence_embed_train, visual_feature_test, sentence_embed_test):
        name="CTRL_Model"
        with tf.variable_scope(name):
            print("Building training network...............................\n")
            transformed_clip_train_mix = fc('v2s_lt', tf.concat([visual_feature_train_pos, visual_feature_train_neg], 0), output_dim=self.semantic_size)
            transformed_clip_train_norm_mix = tf.nn.l2_normalize(transformed_clip_train_mix, dim=1)

            transformed_sentence_train = fc('s2s_lt', sentence_embed_train, output_dim=self.semantic_size)
            transformed_sentence_train_norm = tf.nn.l2_normalize(transformed_sentence_train, dim=1)
            cross_modal_vec_train_mix = self.cross_modal_comb(transformed_clip_train_norm_mix,
                                                              tf.tile(transformed_sentence_train_norm, [2,1]),
                                                              self.batch_size)

            sim_score_mat_train_mix = vs_multilayer.vs_multilayer(cross_modal_vec_train_mix, "vs_multilayer_lt", middle_layer_dim=1000)
            sim_score_mat_train_mix = tf.reshape(sim_score_mat_train_mix, [self.batch_size*2, 3])

            tf.get_variable_scope().reuse_variables()
            print("Building test network...............................\n")
            transformed_clip_test = fc('v2s_lt', visual_feature_test, output_dim=self.semantic_size)
            transformed_clip_test_norm = tf.nn.l2_normalize(transformed_clip_test, dim=1)
            transformed_sentence_test = fc('s2s_lt', sentence_embed_test, output_dim=self.semantic_size)
            transformed_sentence_test_norm = tf.nn.l2_normalize(transformed_sentence_test, dim=1)
            cross_modal_vec_test = self.cross_modal_comb(transformed_clip_test_norm, transformed_sentence_test_norm, self.test_batch_size)
            sim_score_mat_test = vs_multilayer.vs_multilayer(cross_modal_vec_test, "vs_multilayer_lt", reuse=True, middle_layer_dim=1000)
            sim_score_mat_test = tf.reshape(sim_score_mat_test, [3])

            return sim_score_mat_train_mix, sim_score_mat_test

    '''
    compute alignment and regression loss
    '''
    def compute_loss_reg(self, sim_reg_mat_mix, offset_label):
        # sim_reg_mat_pos = sim_reg_mat_mix[:sim_reg_mat_mix.get_shape().as_list()[0]/2]
        # sim_reg_mat_neg = sim_reg_mat_mix[sim_reg_mat_mix.get_shape().as_list()[0]/2:]
        sim_score_mat, _, _ = tf.split(sim_reg_mat_mix, num_or_size_splits=3, axis=1)

        mask_mat = tf.concat((tf.constant(-1.0, shape=[self.batch_size]), tf.constant(1.0, shape=[self.batch_size])), 0)
        all1 = tf.constant(-1.0, shape=[self.batch_size*2])
        loss_mat = tf.log(tf.add(all1, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
        loss_align = tf.reduce_mean(loss_mat)


        # regression loss
        _, p_reg_mat, l_reg_mat = tf.split(sim_reg_mat_mix[:self.batch_size], num_or_size_splits=3, axis=1)
        #I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
        #l_reg_diag = tf.matmul(tf.multiply(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        #p_reg_diag = tf.matmul(tf.multiply(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        offset_pred = tf.concat((p_reg_mat, l_reg_mat), 1)
        loss_reg = tf.reduce_mean(tf.abs(tf.subtract(offset_pred, offset_label)))

        loss=tf.add(tf.multiply(self.lambda_regression, loss_reg), loss_align)
        return loss, offset_pred, loss_reg


    def init_placeholder(self):
        visual_featmap_ph_train_pos = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim))
        visual_featmap_ph_train_neg = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim))
        sentence_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.sentence_embedding_size))
        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size,2))
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim))
        sentence_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))

        return visual_featmap_ph_train_pos, visual_featmap_ph_train_neg, \
               sentence_ph_train,offset_ph, visual_featmap_ph_test, sentence_ph_test
    

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
        self.visual_featmap_ph_train_pos,  self.visual_featmap_ph_train_neg, self.sentence_ph_train, self.offset_ph, \
        self.visual_featmap_ph_test, self.sentence_ph_test=self.init_placeholder()

        # build inference network
        sim_reg_mat_mix, sim_reg_mat_test = self.visual_semantic_infer(self.visual_featmap_ph_train_pos,
                                                                   self.visual_featmap_ph_train_neg,
                                                                   self.sentence_ph_train,
                                                                   self.visual_featmap_ph_test, self.sentence_ph_test)
        # compute loss
        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_reg_mat_mix, self.offset_ph)
        # optimize
        self.vs_train_op = self.training(self.loss_align_reg)
        return self.loss_align_reg, self.vs_train_op, sim_reg_mat_test, offset_pred, loss_reg


