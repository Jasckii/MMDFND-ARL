import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score

def clipdata2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda(),
        'clip_image':batch[5].cuda(),
        'clip_text': batch[6].cuda()
    }
    return batch_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v

# =====================================================================
# 核心指标计算 (支持 Validation 搜索与 Test 盲测)
# =====================================================================
def metrics(y_true, y_pred_prob, category, category_dict, is_test=False, thresholds_dict=None):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {v: k for k, v in category_dict.items()}
    
    for k in category_dict.keys():
        res_by_category[k] = {"y_true": [], "y_pred_prob": [], "indices": []}

    for i, c in enumerate(category):
        c_name = reverse_category_dict[c]
        res_by_category[c_name]['y_true'].append(y_true[i])
        res_by_category[c_name]['y_pred_prob'].append(y_pred_prob[i])
        res_by_category[c_name]['indices'].append(i)

    y_pred_bin_global = np.zeros(len(y_pred_prob), dtype=int)
    
    # 建立一个字典，用来存储这轮搜索到的最佳阈值
    searched_thresholds = {}

    for c, res in res_by_category.items():
        cat_y_true = np.array(res['y_true'])
        cat_y_pred_prob = np.array(res['y_pred_prob'])
        
        try:
            cat_auc = roc_auc_score(cat_y_true, cat_y_pred_prob).round(4).tolist()
        except ValueError:
            cat_auc = 0
            
        # -------------------------------------------------------------
        # 👑 核心逻辑分流：验证集搜索 VS 测试集盲测
        # -------------------------------------------------------------
        if not is_test:
            # 模式 A: Validation (网格搜索最佳阈值)
            best_f1 = 0
            best_cat_thresh = 0.5
            for th in np.arange(0.3, 0.71, 0.05):
                preds_bin = (cat_y_pred_prob >= th).astype(int)
                f1 = f1_score(cat_y_true, preds_bin, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_cat_thresh = round(th, 2)
            # 保存搜索到的最佳阈值
            searched_thresholds[c] = best_cat_thresh
        else:
            # 模式 B: Test (绝对不搜索！直接套用 Validation 传来的阈值)
            if thresholds_dict is not None and c in thresholds_dict:
                best_cat_thresh = thresholds_dict[c]
            else:
                best_cat_thresh = 0.5 # 兜底安全策略
        # -------------------------------------------------------------
        
        # 依照最终确定的阈值进行二值化
        cat_y_pred_bin = (cat_y_pred_prob >= best_cat_thresh).astype(int)
        
        # 填回全局阵列
        for idx, global_idx in enumerate(res['indices']):
            y_pred_bin_global[global_idx] = cat_y_pred_bin[idx]
            
        if len(cat_y_true) > 0:
            metrics_by_category[c] = {
                'precision': precision_score(cat_y_true, cat_y_pred_bin, average='macro', zero_division=0).round(4).tolist(),
                'recall': recall_score(cat_y_true, cat_y_pred_bin, average='macro', zero_division=0).round(4).tolist(),
                'fscore': f1_score(cat_y_true, cat_y_pred_bin, average='macro', zero_division=0).round(4).tolist(),
                'auc': cat_auc,
                'acc': accuracy_score(cat_y_true, cat_y_pred_bin).round(4),
                'best_thresh': best_cat_thresh 
            }
        else:
            metrics_by_category[c] = {'precision': 0, 'recall': 0, 'fscore': 0, 'auc': 0, 'acc': 0, 'best_thresh': best_cat_thresh}

    # 全局指标计算
    try:
        metrics_by_category['auc'] = roc_auc_score(y_true, y_pred_prob, average='macro')
    except ValueError:
        metrics_by_category['auc'] = 0

    metrics_by_category['metric'] = f1_score(y_true, y_pred_bin_global, average='macro', zero_division=0)
    metrics_by_category['recall'] = recall_score(y_true, y_pred_bin_global, average='macro', zero_division=0)
    metrics_by_category['precision'] = precision_score(y_true, y_pred_bin_global, average='macro', zero_division=0)
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred_bin_global)

    # Validation 会返回额外的阈值字典，Test 则只返回结果
    if not is_test:
        return metrics_by_category, y_pred_bin_global, searched_thresholds
    else:
        return metrics_by_category, y_pred_bin_global

# =====================================================================
# 包含真假新闻独立指标的包装函数
# =====================================================================
def metricsTrueFalse(y_true, y_pred_prob, category, category_dict, is_test=False, thresholds_dict=None):
    
    # 动态解包
    if not is_test:
        metrics_res, y_pred_bin_global, searched_thresholds = metrics(y_true, y_pred_prob, category, category_dict, is_test=False)
    else:
        metrics_res, y_pred_bin_global = metrics(y_true, y_pred_prob, category, category_dict, is_test=True, thresholds_dict=thresholds_dict)
        
    y_true_arr = np.array(y_true)
    
    fake = {
        'precision': precision_score(y_true_arr, y_pred_bin_global, pos_label=1, zero_division=0),
        'recall': recall_score(y_true_arr, y_pred_bin_global, pos_label=1, zero_division=0),
        'F1': f1_score(y_true_arr, y_pred_bin_global, pos_label=1, zero_division=0)
    }
    real = {
        'precision': precision_score(y_true_arr, y_pred_bin_global, pos_label=0, zero_division=0),
        'recall': recall_score(y_true_arr, y_pred_bin_global, pos_label=0, zero_division=0),
        'F1': f1_score(y_true_arr, y_pred_bin_global, pos_label=0, zero_division=0)
    }
    metrics_res['real'] = real
    metrics_res['fake'] = fake
    
    # 将寻找出的阈值一并返回给主函数
    if not is_test:
        return metrics_res, searched_thresholds
    else:
        return metrics_res
    
def data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda()
    }
    return batch_data

class Recorder():
    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("Current", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)
