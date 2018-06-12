
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from extract_feature import NET_OUT_CHANNEL
RPN_POS_IOU = 0.7
RPN_NEG_IOU = 0.3
NUM_PRE_NMS = 8000
NUM_AFTER_NMS = 1000
NMS_TRESH = 0.8


# In[3]:


###1.anchor生成模块 产生H/16*W/16个框
###输入：feature map的W、H大小
###输出：W*H个anchor坐标


# In[4]:


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],anchor_scales=[8, 16, 32]):
    py = 0. 
    px = 0. 
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i]) #16 *8*根2/2
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i]) #16*8*根2

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = px - w / 2 
            anchor_base[index, 1] = py - h / 2
            anchor_base[index, 2] = px + w / 2
            anchor_base[index, 3] = py + h / 2
    return anchor_base

def _enumerate_shifted_anchor(anchor_base,width,height,feat_stride):
    #生成w*h*9个的anchor，并标出所有anchor的坐标
    #feat_stride为相邻anchor间隔，VGG feature与原图缩小16倍，所以采用16
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    ####change
    anchor[:, slice(0, 4, 2)] = np.clip(anchor[:, slice(0, 4, 2)], 0, width*16) 
    anchor[:, slice(1, 4, 2)] = np.clip(anchor[:, slice(1, 4, 2)], 0, height*16)  
    return anchor

def generate_img_anchor(width,height,feat_stride=16):
    anchor_base = generate_anchor_base()
    anchor = _enumerate_shifted_anchor(anchor_base,width,height,feat_stride)
    return anchor


# In[6]:


if __name__=='__main__':
    anchor = generate_img_anchor(32,20)
    for i in range(30,34):
        print(anchor[i*9])


# In[4]:


####2、AnchorTargetCreator 选取128个正样本和128个负样本，输出label和转换过的loc信息
###输入：1、所有原图gt_bbox坐标   2、生成的W*H个anchor   3、原图的W1和H1
###输出：1、E=W*H个标注，正样本占128个，负样本占128个   2、转换之后的位置参数  3、正、负样本的索引


# In[5]:


def _get_inside_index(anchor, W, H):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= W) &
        (anchor[:, 3] <= H)
    )[0]
    return index_inside

def bbox_iou(bbox_a, bbox_b):
    #计算bbox_a 与bbox_b的IOU
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def _calc_ious(anchor, gt_bbox, inside_index):
    # ious between the anchors and the gt boxes
    ious = bbox_iou(anchor, gt_bbox)#列为gt box  行为anchor
    argmax_ious = ious.argmax(axis=1)# 每个anchor与所有gt box最大的IOU 列索引
    max_ious = ious[np.arange(len(inside_index)), argmax_ious]# m每行最大值，每个anchor最大的IOU（20000+） 
    gt_argmax_ious = ious.argmax(axis=0) #每个gt box与所有anchor最大的IOU行索引
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]#每列最大值
    gt_argmax_ious = np.where(ious == gt_max_ious)[0] #与每列最大值相同的所有行索引
    return argmax_ious, max_ious, gt_argmax_ious

def _create_label( inside_index, anchor, gt_bbox,
                  pos_iou_thresh = RPN_POS_IOU,neg_iou_thresh = RPN_NEG_IOU,n_sample = 256):
    # label: 1 is positive, 0 is negative, -1 is dont care
    pos_ratio = 0.5
    label = np.empty((len(inside_index),), dtype=np.int32)
    label.fill(-1)

    argmax_ious, max_ious, gt_argmax_ious = _calc_ious(anchor, gt_bbox, inside_index)

    # assign negative labels first so that positive labels can clobber them
    label[max_ious < neg_iou_thresh] = 0

    # positive label: for each gt, anchor with highest iou
    label[gt_argmax_ious] = 1

    # positive label: above threshold IOU
    label[max_ious >= pos_iou_thresh] = 1

    # subsample positive labels if we have too many
    n_pos = int(pos_ratio * n_sample)
    pos_index = np.where(label == 1)[0] 
    if len(pos_index) > n_pos: #选取128个正例子
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1

    ###if len(pos_index)<n_pos:   ###change
    ###    n_neg = np.sum(label == 1) #如果正样例不够128，只选与正相当的负样例
    ###else:
    n_neg = n_sample - np.sum(label == 1)  # subsample negative labels if we have too many
        
    neg_index = np.where(label == 0)[0] #选128个负例子
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1
    return argmax_ious, label

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height
    
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height
    
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

def AnchorTargetCreator(gt_bbox,anchor,img_w,img_h):
    #1:计算gt_bbox与anchor的iou值
    #2:选取与gt_bbox最大的一个为正样本，IOU大于0.7的为正，IOU小于0.3的为负，正负各选128个
    #3:对选出样本的anchor做变化，用于回归
    inside_index = _get_inside_index(anchor,img_w,img_h)
    anchor = anchor[inside_index]
    argmax_ious, train_label = _create_label(inside_index,anchor,gt_bbox)
    train_loc = bbox2loc(anchor, gt_bbox[argmax_ious])
    return train_label,train_loc


# In[6]:


###3。排序，修正anchor位置
###输入：生成的W*H个anchor,卷积之后的label概率，卷积之后得到的loc参数,原图的宽。高
###输出：修正后的60% anchor框


# In[7]:


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]# 对应公式ty、tx、th、tw，求解的是x，y，h，w表示预测的bbox
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

def anchor_update(anchor,conv_label,conv_loc,W,H,scale=1):
    min_size=16
    n_pre_nms=NUM_PRE_NMS
    
    roi = loc2bbox(anchor, conv_loc)
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, W)##将预测box左边超出边界的截取回到图像内
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, H)
    
    min_size = min_size * scale
    ws = roi[:, 2] - roi[:, 0]
    hs = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0] #返回宽高大于min_size的roi区域
    roi = roi[keep, :]
    conv_label = conv_label[keep]
    
    order = conv_label.ravel().argsort()[::-1]
    order = order[:n_pre_nms]
    anchor_to_nms = roi[order, :]
    label_to_nms = conv_label[order]
    return anchor_to_nms,label_to_nms


# In[8]:


###4、NMS筛选ROI区域
###输入: 调整坐标之后的6000个anchor
###输出：300个ROI区域


# In[9]:


def py_cpu_nms(anchor, label,thresh=NMS_TRESH):    
    x1 = anchor[:, 0]  
    y1 = anchor[:, 1]  
    x2 = anchor[:, 2]  
    y2 = anchor[:, 3]  
    scores = label[:]  
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)   
    order = scores.argsort()[::-1] #label从大到小的索引值  
    keep = []  
    while order.size > 0:  
        i = order[0]   #第一个计算值
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h    
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]   
        order = order[inds + 1]  
  
    return keep  


# In[10]:


###5、RPN训练过程(损失函数定义)
###输入: 卷积层的feature map,图像宽高
###输出：300个ROI区域 
###过程：
###1、生成20000个anchor
###2、训练阶段：根据gt_loc信息，标记128个正样本和128个负样本，用正样本做坐标训练，用正负样本做分类训练。计算损失，反向传播
###3、卷积得到label和loc的信息，选出label前6000的anchor
###4、通过nms选出300个anchor


# In[11]:


def Region_Proposal_Network(label_numpy,loc_numpy,img_W,img_H,feat_stride=16):
    ####1、生成anchors
    F_H = (img_H+feat_stride-1)//feat_stride
    F_W = (img_W+feat_stride-1)//feat_stride
    anchor = generate_img_anchor(F_W,F_H,feat_stride=feat_stride)
    label_numpy = label_numpy[:,1]#正样本概率
    anchor_to_nms,label_to_nms = anchor_update(anchor,label_numpy,loc_numpy,img_W,img_H) 
    ###4、选出12000个丢进nms选出300个
    keep   = py_cpu_nms(anchor_to_nms, label_to_nms)
    keep  = keep[:NUM_AFTER_NMS]
    bbox_to_roi = anchor_to_nms[keep]
    return bbox_to_roi,anchor

def Region_Proposal_Network_test(label_numpy,loc_numpy,img_W,img_H,feat_stride=16):
    ####1、生成anchors
    F_H = (img_H+feat_stride-1)//feat_stride
    F_W = (img_W+feat_stride-1)//feat_stride
    anchor = generate_img_anchor(F_W,F_H,feat_stride=feat_stride)
    label_numpy = label_numpy[:,1]#正样本概率
    anchor_to_nms,label_to_nms = anchor_update(anchor,label_numpy,loc_numpy,img_W,img_H) 
    ###4、选出12000个丢进nms选出300个
    keep   = py_cpu_nms(anchor_to_nms, label_to_nms)
    label_to_nms = label_to_nms[keep]
    bbox_to_roi = anchor_to_nms[keep]
    order = label_to_nms.ravel().argsort()[::-1]
    order  = order[:64]
    bbox_to_roi = bbox_to_roi[order]
    return bbox_to_roi

