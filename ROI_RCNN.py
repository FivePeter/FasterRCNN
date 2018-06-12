
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from Region_Proposal_Network import bbox_iou
from Region_Proposal_Network import bbox2loc
from Region_Proposal_Network import loc2bbox
from extract_feature import NET_OUT_CHANNEL
ROI_POS_THR = 0.5
ROI_NEG_THR = 0.1
ROI_SMAPLE_NUM =64
ROI_SIZE = 7
NET_OUT_CHANNEL = 512


# In[ ]:


VERBOSE = 0


# In[ ]:


###输入：送入经过ProposalTargetCreator筛选的ROI
###输出：ROI pooling之后的feature
###过程：1、计算所有ROI映射到feature 上的feature'，
###2、feature'经过RIOPooling成7x7的区域
###3、合并所有ROI pooling之后的图（batch_size不为1）
def ROIPooling(out_W,out_H,feature,rois,spatial_scale=1/16):
    roi = tf.to_int32(rois*spatial_scale)
    roi_num = ROI_SMAPLE_NUM
    size = [out_H,out_W]
    for i in range(roi_num):
        roi_feature = feature[:,roi[i,1]:(roi[i,3]+1),roi[i,0]:(roi[i,2]+1),:] ##B H W C
        mm = tf.image.resize_images(roi_feature,size,method=0) ####代替ROI pooling过程
        if i==0:
            output = mm
        else:
            output = tf.concat([mm,output],0)
    return output

###输入:RPN选出的roi, 原图的gt_bbox 原图的gt_label
###输出：(1)sample_roi：选出的ROI区域原图坐标 
####(2)gt_roi_loc: ROI转换为loc信息，并标准化，用于作为回归的标注数据  (3)gt_roi_label：标注的label信息，用作分类的标注数据
def ProposalTargetCreator(roi,gt_bbox,gt_label,n_sample=ROI_SMAPLE_NUM,pos_ratio=0.5,
            loc_normalize_mean=(0., 0., 0., 0.),loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            pos_iou_thresh = ROI_POS_THR,neg_iou_thresh_hi = ROI_POS_THR):
    neg_iou_thresh_lo = ROI_NEG_THR
    n_bbox, _ = gt_bbox.shape
    
    roi = np.concatenate((roi, gt_bbox), axis=0) #将

    pos_roi_per_image = np.round(n_sample * pos_ratio) #四舍五入32正样本
    iou = bbox_iou(roi, gt_bbox)   #计算IOU
    if VERBOSE: print('iou shape is ',iou.shape)
    gt_assignment = iou.argmax(axis=1) #每一列最大值行索引
    max_iou = iou.max(axis=1)#每个roi重合最大的一个gt_bbox
    gt_roi_label = gt_label[gt_assignment] + 1 #新增0表示负样本，其余label往上加1
    if VERBOSE: print('max_iou is ',max_iou)

    pos_index = np.where(max_iou >= pos_iou_thresh)[0] #
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

    neg_index = np.where((max_iou < neg_iou_thresh_hi)&(max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image,neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

    keep_index = np.append(pos_index, neg_index)
    if VERBOSE: print('chose index is ',keep_index)
    if VERBOSE: print('pos sample num is',pos_roi_per_this_image,'neg sample num is',neg_roi_per_this_image)
    gt_roi_label = gt_roi_label[keep_index]
    gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
    sample_roi = roi[keep_index]
    
    gt_roi_loc = bbox2loc(sample_roi, gt_bbox[gt_assignment[keep_index]])
    if VERBOSE: print('gt_roi_loc before normalization is ',gt_roi_loc)
    #gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))
    if keep_index.shape[0]!=64:
        print("error roi img batch is not 64. neg is",neg_index.shape[0],"pos is",pos_index.shape[0])
    return sample_roi, gt_roi_loc, gt_roi_label


# In[ ]:


####输入：300个修正过的roi,gt_bbox,gt_label
####输出：预测的label,bbox
####过程: 1、从输入的300个anchor挑选 32正  96负
####2、对选出的128个loc信息标准化处理
####3、根据128个anchor，得到128个feature（原图映射到feature上）
####4、分别将128个feature 通过ROI pooling到7x7
####5、两个4096x4096的FC层，一个Fc21用于分类，一个FC 84用于回顾
####6、与标注的loc和label计算损失函数（只在训练阶段使用）
####7、修正坐标，选出概率最高的一类作为一个feature的输出
####8、设置大于某个概率值得框作为预测输出


# In[ ]:


def VGG16RoIHead(sample_roi,feature):
    fc_input = ROIPooling(ROI_SIZE,ROI_SIZE,feature,sample_roi)
    pool_node0 = ROI_SMAPLE_NUM
    pool_node1 = ROI_SIZE
    pool_node2 = ROI_SIZE
    pool_node3 = NET_OUT_CHANNEL
    nodes = pool_node1*pool_node2*pool_node3
    fc_input = tf.reshape(fc_input, [pool_node0, nodes])
    
    fc1_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[nodes, 3072],stddev=0.01,dtype=tf.float32),
                              name="fc1_weights")
    fc2_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[3072, 3072],stddev=0.01,dtype=tf.float32),
                              name="fc2_weights")
    fc_cls_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[3072, 21],stddev=0.01,dtype=tf.float32),
                                 name="fc_cls_weights")
    fc_rgr_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[3072, 84],stddev=0.01,dtype=tf.float32),
                                 name="fc_rgr_weights")

    fc1_biases = tf.Variable(initial_value=tf.constant(0,shape=[3072],dtype=tf.float32),name="fc_bias1")
    fc2_biases = tf.Variable(initial_value=tf.constant(0,shape=[3072],dtype=tf.float32),name="fc_bias2")
    fc_cls_biases = tf.Variable(initial_value=tf.constant(0,shape=[21],dtype=tf.float32),name="bias_roi_cls")
    fc_rgr_biases = tf.Variable(initial_value=tf.constant(0,shape=[84],dtype=tf.float32),name="bias_roi_rgr")
    
    fc1 = tf.nn.relu(tf.matmul(fc_input, fc1_weights) + fc1_biases)
    fc1 = tf.nn.dropout(fc1, 0.5)
    
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    fc2 = tf.nn.dropout(fc2, 0.5)
        
    fc_cls_21 = tf.nn.relu(tf.matmul(fc2, fc_cls_weights) + fc_cls_biases)
    fc_rgr_84 = tf.nn.relu(tf.matmul(fc2, fc_rgr_weights) + fc_rgr_biases)
    
    return fc_cls_21,fc_rgr_84

