import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import torch.nn.functional as F

# --- 【ARL 1/3】: 梯度缩放层 ---
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # 前向传播不做修改，但必须保存 weight 供反向传播使用
        ctx.save_for_backward(weight)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 取出前向传播保存的权重
        weight = ctx.saved_tensors[0]
        # 【核心逻辑】：梯度 = 原始梯度 * (1 + weight)
        # weight 越大，该模态获得的梯度越大，学习越快
        grad_input = grad_output + grad_output * weight
        # weight 本身不需要梯度，所以返回 None
        return grad_input, None
# --------------------------------------

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(MultiDomainPLEFENDModel, self).__init__()
        
        # --- 【ARL 新增代码 2/3 - Part A】: 初始化 ARL 参数 ---
        # 使用 register_buffer 确保这些参数随模型保存/移动到GPU，但不是可训练参数
        self.register_buffer('text_weight', torch.tensor(0.0))
        self.register_buffer('image_weight', torch.tensor(0.0))
        
        # ARL 超参数设定
        self.arl_start_epoch = 2 # 默认第n轮后才开始 ARL，前期让模型自由预热
        self.current_epoch = 0   # 记录当前轮次
        # --------------------------------------------------
        
        self.num_expert = 6
        self.domain_num = 9
        self.gate_num = 10
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        #self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        self.bert = BertModel.from_pretrained(bert)
        
        # 开启 BERT 的梯度检查点，极大地节省显存
        self.bert.gradient_checkpointing_enable()
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
                #image_expert.append(image_cnn_extractor())
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(160),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert,final_share_expert = [], [], [],[]
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert*2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                #image_share.append(image_cnn_extractor())
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(160),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0,final_gate_list = [], [], [], [],[]
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                      nn.SiLU(),
                                      #SimpleGate(),
                                      #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(self.unified_dim, self.num_expert * 4),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         #SimpleGate(),
                                         #nn.BatchNorm1d(80),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(1088, 720),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(720, 160),
                                        nn.SiLU(),
                                        nn.Linear(160, self.num_expert * 3),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                         )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        #self.text_attention = TokenAttention(self.unified_dim)
        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)

        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)

        text_classifier_list = []

        for i in range(self.domain_num):
            text_classifier = MLP(320, mlp_dims, dropout)
            text_classifier_list.append(text_classifier)
        self.text_classifier_list = nn.ModuleList(text_classifier_list)

        image_classifier_list = []

        for i in range(self.domain_num):
            image_classifier = MLP(320, mlp_dims, dropout)
            image_classifier_list.append(image_classifier)
        self.image_classifier_list = nn.ModuleList(image_classifier_list)

        fusion_classifier_list = []

        for i in range(self.domain_num):
            fusion_classifier = MLP(320, mlp_dims, dropout)
            fusion_classifier_list.append(fusion_classifier)
        self.fusion_classifier_list = nn.ModuleList(fusion_classifier_list)

        share_classifier_list = []

        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []

        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)



        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(1088, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)


        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        #for param in self.image_model.parameters():
            #param.requires_grad = False

        #### mapping MLPs
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim,1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()
        self.irrelevant_tensor = []
        for i in range(self.domain_num):
            self.irrelevant_tensor.append(nn.Parameter(torch.ones((1, 320)), requires_grad=True))
        
        # 原始代码
        self.ClipModel,_ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        
        # 修改为：
        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        self.ClipModel = self.ClipModel.float() # 强制转换为 float32 以支持微调


        #pivot:
        feature_emb_size = 320
        img_emb_size =320
        feature_num = 4
        self.feature_num = 4
        text_emb_size = 320
        #self.n_node = 64
        self.feature_emb_size = 320
        self.emb_size = 320
        self.layers = 12
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)])
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])
        self.pivot_mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        self.transformers_list = torch.nn.ModuleList()
        self.mlp_img_list = torch.nn.ModuleList()
        self.mlp_text_list = torch.nn.ModuleList()
        self.pivot_mlp_fusion_list = torch.nn.ModuleList()
        for i in range(self.domain_num):
            self.transformers_list.append(torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)]))
            self.mlp_img_list.append(torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)]))
            self.mlp_text_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)]))
            self.pivot_mlp_fusion_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)]))


        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)
        self.mlp_star_f1 = nn.Linear(self.feature_emb_size * 4, self.emb_size)
        self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_star_f1_list = torch.nn.ModuleList()
        self.mlp_star_f2_list = torch.nn.ModuleList()
        for i in range(self.domain_num):
            self.mlp_star_f1_list.append(nn.Linear(self.feature_emb_size * 4, self.emb_size))
            self.mlp_star_f2_list.append(nn.Linear(self.emb_size, self.emb_size))
            
    #融合----------------------------------------------------
    def fusion_img_text(self, image_emb, text_emb,fusion_emb,mlp_img,mlp_text,mlp_fusion,transformers,mlp_star_f1,mlp_star_f2):
        for img_feature_num in range(0, self.feature_num):
            if img_feature_num == 0:
                img_feature_seq = mlp_img[img_feature_num](image_emb)
                img_feature_seq = img_feature_seq.unsqueeze(1)
            else:
                img_feature_seq = torch.cat((img_feature_seq, mlp_img[img_feature_num](image_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                text_feature_seq = mlp_text[text_feature_num](text_emb)
                text_feature_seq = text_feature_seq.unsqueeze(1)
            else:
                text_feature_seq = torch.cat((text_feature_seq, mlp_text[text_feature_num](text_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                fusion_feature_seq = mlp_fusion[text_feature_num](fusion_emb)
                fusion_feature_seq = fusion_feature_seq.unsqueeze(1)
            else:
                fusion_feature_seq = torch.cat((fusion_feature_seq, mlp_fusion[text_feature_num](fusion_emb).unsqueeze(1)), 1)
        #print(img_feature_seq.shape)
        #print(text_feature_seq.shape)
        #print(fusion_feature_seq.shape)
        #star_emb1 = (img_feature_seq[:, 0, :] + text_feature_seq[:, 0, :] + fusion_feature_seq[:, 0, :]) / 3
        #star_emb2 = (img_feature_seq[:, 1, :] + text_feature_seq[:, 1, :] + fusion_feature_seq[:, 1, :]) / 3
        #star_emb3 = (img_feature_seq[:, 2, :] + text_feature_seq[:, 2, :] + fusion_feature_seq[:, 2, :]) / 3
        #star_emb4 = (img_feature_seq[:, 3, :] + text_feature_seq[:, 3, :] + fusion_feature_seq[:, 3, :]) / 3
        star_emb1 = text_feature_seq[:, 0, :]
        star_emb2 = text_feature_seq[:, 1, :]
        star_emb3 = text_feature_seq[:, 2, :]
        star_emb4 = text_feature_seq[:, 3, :]




        for sa_i in range(0, int(self.layers), 3):
            trans_text_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1), star_emb4.unsqueeze(1),  text_feature_seq], 1)
            text_output = transformers[sa_i + 2](trans_text_item)

            star_emb1 = (text_output[:, 0, :] + star_emb1)/2
            star_emb2 = (text_output[:, 1, :] + star_emb2)/2
            star_emb3 = (text_output[:, 2, :] + star_emb3)/2
            star_emb4 = (text_output[:, 3, :] + star_emb4)/2
            text_feature_seq = text_output[:, 4:self.feature_num+4, :] + text_feature_seq

            trans_img_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 img_feature_seq], 1)
            img_output = transformers[sa_i+1](trans_img_item)
            star_emb1 = (img_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (img_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (img_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (img_output[:, 3, :] + star_emb4) / 2
            img_feature_seq = img_output[:, 4:self.feature_num + 4, :] + img_feature_seq

            trans_fusion_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 fusion_feature_seq], 1)
            fusion_output = transformers[sa_i](trans_fusion_item)
            star_emb1 = (fusion_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (fusion_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (fusion_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (fusion_output[:, 3, :] + star_emb4) / 2
            fusion_feature_seq = fusion_output[:, 4:self.feature_num + 4, :] + fusion_feature_seq

        item_emb_trans = self.dropout2(torch.cat([star_emb1, star_emb2, star_emb3,star_emb4], 1))
        item_emb_trans = self.dropout2(self.active(mlp_star_f1(item_emb_trans)))
        item_emb_trans = self.dropout2(self.active(mlp_star_f2(item_emb_trans)))
        return item_emb_trans

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        #image_feature = self.bert(inputs, attention_mask=masks)[0]
        
        # --- 【ARL 新增代码 2/3 - Part B】: 应用梯度调制 ---
        # 只有在训练模式且达到启动轮次后才应用
        if self.training and self.current_epoch >= self.arl_start_epoch:
            text_feature = GradScale.apply(text_feature, self.text_weight)
            image_feature = GradScale.apply(image_feature, self.image_weight)
        # ------------------------------------------------
        # ---------------- 原始代码 ----------------
        #clip_image = kwargs['clip_image']
        #clip_text = kwargs['clip_text']
        #with torch.no_grad():
            #clip_image_feature = self.ClipModel.encode_image(clip_image)# ([64, 512])
            #clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            #clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            #clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
            ##print(clip_image_feature.size())
            ##print(clip_text_feature.size())
        #clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature),dim=-1)#torch.Size([64, 1024])
        #clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())#torch.Size([64, 320])
        # ------------------------------------------
        
        # ================ 修改为以下代码 ================
        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        
        # 1. 移除 with torch.no_grad():，让计算图包含 CLIP
        clip_image_feature = self.ClipModel.encode_image(clip_image)  # ([64, 512])
        clip_text_feature = self.ClipModel.encode_text(clip_text)     # ([64, 512])
        
        # 2. 归一化特征
        clip_image_feature = clip_image_feature / clip_image_feature.norm(dim=-1, keepdim=True)
        clip_text_feature = clip_text_feature / clip_text_feature.norm(dim=-1, keepdim=True)

        # 3. 【核心新增】：复用 ARL 权重对 CLIP 的图/文单边特征进行梯度调制
        if self.training and self.current_epoch >= self.arl_start_epoch:
            clip_text_feature = GradScale.apply(clip_text_feature, self.text_weight)
            clip_image_feature = GradScale.apply(clip_image_feature, self.image_weight)

        # 4. 拼接并输入融合层
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())
        # ================================================
        
        #text_atn_feature, _ = self.text_attention(text_feature)  # ([64, 768])
        text_atn_feature = self.text_attention(text_feature,masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)
        # print("image_atn_feature", image_atn_feature.size())

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  ##([32, 768])
        text_gate_input = torch.cat([domain_embedding, text_atn_feature], dim=-1)  # ([64, 1536])
        image_gate_input = torch.cat([domain_embedding, image_atn_feature], dim=-1)
        fusion_gate_input = torch.cat([domain_embedding, fusion_atn_feature], dim=-1)

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list
        # self.text_gate_out_list = nn.ModuleList(text_gate_out_list)

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list


        # text
        text_gate_expert_value = []
        text_gate_spacial_expert_value = []
        text_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            #print("gate_expert",gate_expert.size()) ([64, 320])
            text_gate_expert_value.append(gate_expert)
            text_gate_spacial_expert_value.append(gate_spacial_expert)
            text_gate_share_expert_value.append(gate_share_expert)

        image_gate_expert_value = []
        image_gate_spacial_expert_value = []
        image_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            image_gate_expert_value.append(gate_expert)
            image_gate_spacial_expert_value.append(gate_spacial_expert)
            image_gate_share_expert_value.append(gate_share_expert)

        #clip_fusion_feature
        #fusion

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((clip_fusion_feature,text, image), dim=-1)

        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = clip_fusion_feature
        fusion_gate_input0 = self.domain_fusion(torch.cat([domain_embedding, fusion_share_feature], dim=-1))
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0


        fusion_gate_expert_value0 = []
        fusion_gate_spacial_expert_value0 = []
        fusion_gate_share_expert_value0 = []
        for m in range(self.domain_num):
            share_gate_expert0 = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
                gate_spacial_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_expert_value0.append(share_gate_expert0)
            fusion_gate_spacial_expert_value0.append(gate_spacial_expert)
            fusion_gate_share_expert_value0.append(gate_share_expert)

##continue

        #test
        text_only_output = []
        text_label_pred = []
        final_text_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_text_feature.append(text_gate_expert_value[i])
            text_class = self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)
            text_only_output.append(text_class)
            pre = torch.sigmoid(text_class)
            text_label_pred.append(pre)
        text_label_pred_list = []
        text_label_pred_avg = 0
        for i in range(self.domain_num):
            text_label_pred_list.append(text_label_pred[i][idxs.squeeze() == i])
            text_label_pred_avg += text_label_pred[i]
        text_label_pred_avg = text_label_pred_avg / 8
        text_label_pred_list = torch.cat((text_label_pred_list[0], text_label_pred_list[1], text_label_pred_list[2], text_label_pred_list[3],
                                     text_label_pred_list[4], text_label_pred_list[5], text_label_pred_list[6], text_label_pred_list[7], text_label_pred_list[8]))
        #image
        image_only_output = []
        image_label_pred = []
        final_image_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_image_feature.append(image_gate_expert_value[i])
            image_class = self.image_classifier_list[i](image_gate_expert_value[i]).squeeze(1)
            image_only_output.append(image_class)
            pre = torch.sigmoid(image_class)
            image_label_pred.append(pre)
        image_label_pred_list = []
        image_label_pred_avg = 0
        for i in range(self.domain_num):
            image_label_pred_list.append(image_label_pred[i][idxs.squeeze() == i])
            image_label_pred_avg += image_label_pred[i]
        image_label_pred_avg = image_label_pred_avg / 8

        image_label_pred_list = torch.cat((image_label_pred_list[0], image_label_pred_list[1], image_label_pred_list[2], image_label_pred_list[3],
                                     image_label_pred_list[4], image_label_pred_list[5], image_label_pred_list[6], image_label_pred_list[7], image_label_pred_list[8]))
        # fusion
        fusion_only_output = []
        fusion_label_pred = []
        final_fusion_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_fusion_feature.append(fusion_gate_expert_value0[i])
            fusion_class = self.fusion_classifier_list[i](fusion_gate_expert_value0[i]).squeeze(1)
            fusion_only_output.append(fusion_class)
            pre = torch.sigmoid(fusion_class)
            fusion_label_pred.append(pre)
        fusion_label_pred_list = []
        fusion_label_pred_avg = 0
        for i in range(self.domain_num):
            fusion_label_pred_list.append(fusion_label_pred[i][idxs.squeeze() == i])
            fusion_label_pred_avg += fusion_label_pred[i]
        fusion_label_pred_avg = fusion_label_pred_avg / 9
        fusion_label_pred_list = torch.cat(
            (fusion_label_pred_list[0], fusion_label_pred_list[1], fusion_label_pred_list[2], fusion_label_pred_list[3],
             fusion_label_pred_list[4], fusion_label_pred_list[5], fusion_label_pred_list[6],
             fusion_label_pred_list[7],fusion_label_pred_list[8]))
        # pivot fusion
        text_gate_share_expert_value = text_gate_share_expert_value[0]
        image_gate_share_expert_value = image_gate_share_expert_value[0]
        fusion_gate_share_expert_value = fusion_gate_share_expert_value0[0]
        cross_knowledge = self.fusion_img_text(image_gate_share_expert_value, text_gate_share_expert_value, fusion_gate_share_expert_value,self.mlp_img,self.mlp_text,self.pivot_mlp_fusion,self.transformers,self.mlp_star_f1,self.mlp_star_f2)
        domain_special_list = []
        for i in range(self.domain_num):
            text_spacial_knowledge = text_gate_spacial_expert_value[i]
            image_spacial_knowledge = image_gate_spacial_expert_value[i]
            fusion_spacial_knowledge = fusion_gate_spacial_expert_value0[i]
            domain_knowledge = self.fusion_img_text(image_spacial_knowledge, text_spacial_knowledge,
                                                   fusion_spacial_knowledge,self.mlp_img_list[i],self.mlp_text_list[i],self.pivot_mlp_fusion_list[i],self.transformers_list[i],self.mlp_star_f1_list[i],self.mlp_star_f2_list[i])

            domain_special_list.append(domain_knowledge)
        dom_mu = []
        share_mu = []
        dom_sigma = []
        share_sigma = []
        dom_score_list = []
        share_score_list = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            image_class = self.dom_classifier_list[i](domain_special_list[i]).squeeze(1)
            dom_score_list.append(image_class)
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            image_class = self.share_classifier_list[i](cross_knowledge).squeeze(1)
            share_score_list.append(image_class)

        for i in range(self.domain_num):
            dom_mu.append(self.mapping_IS_MLP_mu(torch.sigmoid(dom_score_list[i]).clone().detach().view(-1,1)))
            share_mu.append(self.mapping_T_MLP_mu(torch.sigmoid(share_score_list[i]).clone().detach().view(-1,1)))
            dom_sigma.append(self.mapping_IS_MLP_sigma(torch.sigmoid(dom_score_list[i]).clone().detach().view(-1,1)))
            share_sigma.append(self.mapping_T_MLP_sigma(torch.sigmoid(share_score_list[i]).clone().detach().view(-1,1)))

        concat_feature_list = []
        for i in range(self.domain_num):
            final_dom_feature0 = self.adaIN(domain_special_list[i],dom_mu[i],dom_sigma[i])
            final_share_feature0 = self.adaIN(cross_knowledge, share_mu[i],share_sigma[i])
            concat_feature_main_biased = torch.stack((final_dom_feature0,
                                                      final_share_feature0,
                                                      ), dim=1)#([64, 2, 320])
            concat_feature_list.append(concat_feature_main_biased)
        final_gate_out_list = []
        for i in range(self.domain_num):
            fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_list[i])
            final_gate_input = torch.cat([domain_embedding, fusion_tempfeat_main_task], dim=-1)
            final_gate_out = self.final_gate_list[i](final_gate_input)
            final_gate_out_list.append(final_gate_out)

        final_gate_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.final_experts[i][j](concat_feature_list[i])  # [64, 4, 320]
                tmp_expert = tmp_expert[:,0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
            for j in range(self.num_expert*2):
                tmp_expert = self.final_share_expert[0][j](concat_feature_list[i])
                tmp_expert = tmp_expert[:, 0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            final_gate_expert_value.append(gate_expert)

        #final
        final_label_pred = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            pre = torch.sigmoid(self.final_classifier_list[i](final_gate_expert_value[i]).squeeze(1))
            final_label_pred.append(pre)
        final_label_pred_list = []
        final_label_pred_avg = 0
        for i in range(self.domain_num):
            final_label_pred_list.append(final_label_pred[i][idxs.squeeze() == i])
            final_label_pred_avg += final_label_pred[i]
        final_label_pred_avg = final_label_pred_avg / 9
        final_label_pred_list = torch.cat((final_label_pred_list[0], final_label_pred_list[1], final_label_pred_list[2], final_label_pred_list[3],
                                     final_label_pred_list[4], final_label_pred_list[5], final_label_pred_list[6], final_label_pred_list[7],final_label_pred_list[8]))



        #return final_label_pred_list, final_label_pred_avg,fusion_label_pred_list, fusion_label_pred_avg,image_label_pred_list, image_label_pred_avg,text_label_pred_list, text_label_pred_avg
        return final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100,
                 # --- [修改片段 6 新增]: 接收外部參數 ---
                 arl_gamma=0.6,
                 arl_T=2.0,
                 pkl_name='parameter_mmdfnd.pkl'
                 # ---------------------------------
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda
        # --- [修改片段 7 新增]: 綁定到實例變數 ---
        self.arl_gamma = arl_gamma
        self.arl_T = arl_T
        self.pkl_name = pkl_name
        # -----------------------------------
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

    def train(self):
        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        
        #optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # =================【初步修改开始：差分学习率优化器】=================
        # 1. 获取 BERT 和 Image Model 的参数 ID
        #bert_params = list(map(id, self.model.bert.parameters()))
        #mae_params = list(map(id, self.model.image_model.parameters()))
        #backbone_ids = bert_params + mae_params
        
        # 2. 筛选出“非骨干”参数（即分类头、门控、Attention层等）
        # 这些参数需要大 LR (self.lr)
        #base_params = filter(lambda p: id(p) not in backbone_ids, self.model.parameters())
        
        # 3. 定义分组优化器
        #optimizer = torch.optim.Adam([
        #    # A组：新添加的层（MLP, Gate, Transformer等），保持原速
            #{'params': base_params, 'lr': self.lr, 'weight_decay': self.weight_decay}, 
            
            # B组：BERT 骨干，使用极小的学习率 (建议 5e-5 或 2e-5)
            #{'params': self.model.bert.parameters(), 'lr': 5e-6,'weight_decay': 1e-2},
            
            # C组：MAE 图像骨干，同样使用极小学习率
            #{'params': self.model.image_model.parameters(), 'lr': 1e-5,'weight_decay': 1e-2}
        #])
        # =================【初步修改结束】=================
        
        # ================ 修改为以下代码 ================
        # 1. 将 CLIP 也标记为需要保护的骨干网络
        bert_params = list(map(id, self.model.bert.parameters()))
        mae_params = list(map(id, self.model.image_model.parameters()))
        clip_params = list(map(id, self.model.ClipModel.parameters())) 
        backbone_ids = bert_params + mae_params + clip_params
        
        base_params = filter(lambda p: id(p) not in backbone_ids, self.model.parameters())
        
        # 2. 将 CLIP 添加到优化器，给予极小的学习率
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': self.lr, 'weight_decay': self.weight_decay}, 
            {'params': self.model.bert.parameters(), 'lr': 5e-6, 'weight_decay': 1e-2},
            {'params': self.model.image_model.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
            # 【新增】：为 CLIP 设置 1e-6 到 5e-6 的学习率
            {'params': self.model.ClipModel.parameters(), 'lr': 1e-6, 'weight_decay': 1e-2}
        ])
        # ================================================
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        
        # --- [修改片段 8 修改]: 替換原本寫死的變數 ---
        arl_T = self.arl_T 
        arl_gamma = self.arl_gamma
        # ----------------------------------------
        
        for epoch in range(self.epoches):
            self.model.train()
            
            # 更新模型内部的 epoch 计数器，用于判断是否开启梯度调制
            self.model.current_epoch = epoch
            
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                category = batch_data['category']
                idxs = torch.tensor([index for index in category]).view(-1, 1)
                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7],label[idxs.squeeze() == 8]))

                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list = self.model(**batch_data)
                
    
                # --- ARL: 动态权重计算逻辑 ---
                if epoch >= self.model.arl_start_epoch:
                    with torch.no_grad(): # 计算权重过程不需要梯度
                        
                        # A. 获取两个模态的预测概率 (进行极小值截断防止 NaN)
                        p_text = torch.clamp(text_label_pred_list, 1e-6, 1-1e-6)
                        p_image = torch.clamp(image_label_pred_list, 1e-6, 1-1e-6)
                        
                        # B. 计算二元熵 (Entropy) - 衡量"不确定性"
                        # 公式: H(p) = -p*log(p) - (1-p)*log(1-p)
                        H_text = - (p_text * torch.log(p_text) + (1-p_text) * torch.log(1-p_text))
                        H_image = - (p_image * torch.log(p_image) + (1-p_image) * torch.log(1-p_image))
                        
                        # 取 batch 均值
                        H_t_mean = H_text.mean().item()
                        H_i_mean = H_image.mean().item()
                        
                        # C. 计算可靠性 (Reliability) - 加入截断保护
                        # 1. 计算相对熵占比 (Relative Entropy Ratio)
                        sum_H = H_t_mean + H_i_mean + 1e-8
                        h_t_ratio = H_t_mean / sum_H
                        h_i_ratio = H_i_mean / sum_H

                        # 2. 【关键】强制截断 (Clamping)
                        # ARL 源码逻辑: H_a = clamp(H_a_n, 0.3)
                        # 作用：无论一个模态表现多好(熵多低)，其在权重计算中的"不确定性占比"最低只能是 0.3
                        # 这防止了权重差异被拉大到无穷大 (例如防止出现 0.01 vs 0.99)
                        h_t_ratio = max(0.3, h_t_ratio)
                        h_i_ratio = max(0.3, h_i_ratio)

                        # 3. 计算可靠性 (熵占比的倒数)
                        # 占比越小(越接近0.3) -> 倒数越大 -> 权重越高
                        r_text = 1.0 / h_t_ratio
                        r_image = 1.0 / h_i_ratio
                        
                        # 4. 归一化可靠性权重
                        r_sum = r_text + r_image
                        w_t_norm = r_text / r_sum
                        w_i_norm = r_image / r_sum
                        
                        # D. 计算置信度 (Confidence) - 预测值的均值幅度
                        conf_text = p_text.mean().item()
                        conf_image = p_image.mean().item()
                        
                        # E. 【ARL 核心非对称交叉加权】
                        # 文本的梯度权重 = 图像的置信度 * 文本自身的可靠性 * 温度系数
                        # 逻辑：利用强模态(高置信度)来指导可靠模态的学习
                        logit_text = (conf_image * w_t_norm) * arl_T
                        logit_image = (conf_text * w_i_norm) * arl_T
                        
                        # F. Softmax 归一化得到最终调制系数
                        weights = F.softmax(torch.tensor([logit_text, logit_image]).cuda(), dim=0)
                        
                        # G. 更新到模型中 (将在下一个 Batch 的 GradScale 中生效)
                        self.model.text_weight = weights[0]
                        self.model.image_weight = weights[1]
                        
                        # 每 50 个 step 打印一次日志 (类似 ARL 源码)
                        if step_n % 50 == 0:
                            # 1. 计算当前 Batch 的简单准确率 (用于监控)
                            # MMDFND 是二分类/多标签任务，阈值取 0.5
                            acc_text = ((text_label_pred_list > 0.5).float() == batch_label.float()).float().mean().item()
                            acc_image = ((image_label_pred_list > 0.5).float() == batch_label.float()).float().mean().item()
                            
                            print(f"\n[ARL Monitor] Epoch {epoch} Step {step_n}")
                            print(f"  > Accuracy  | Text: {acc_text:.4f} | Image: {acc_image:.4f}")
                            print(f"  > Entropy   | Text: {H_t_mean:.4f} | Image: {H_i_mean:.4f} (Larger = More Uncertain)")
                            print(f"  > Confidence| Text: {conf_text:.4f} | Image: {conf_image:.4f}")
                            print(f"  > Reliability Weights (Variance-based) | Text: {w_t_norm:.4f} | Image: {w_i_norm:.4f}")
                            print(f"  > Final Gradient Weights (Softmax)     | Text: {weights[0].item():.4f} | Image: {weights[1].item():.4f}")
                            print("-" * 60)
                #-------------------------------原损失逻辑--------------------------------
                loss0 = loss_fn(final_label_pred_list, batch_label.float())
                loss1 = loss_fn(fusion_label_pred_list, batch_label.float())
                loss2 = loss_fn(image_label_pred_list, batch_label.float())
                loss3 = loss_fn(text_label_pred_list, batch_label.float())
                #loss = 0.7*loss0+0.1*loss1+0.1*loss2+0.1*loss3
                #---------------------------------------------------------------
                loss_main = 0.7 * loss0 + 0.1 * loss1 # 融合相关的 Loss
                loss_aux  = loss2 + loss3             # 单模态 Loss (Image + Text)
                
                # 总 Loss = 主 Loss + gamma * 单模态 Loss
                loss = loss_main + arl_gamma * loss_aux
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results0, results1, results2, results3 = self.test(self.val_loader)
            
            # 【重要修改：措施一】验证集跑完后，立即强制清理显存
            # 这能释放验证集产生的巨大临时 Tensor，防止下一轮 OOM
            torch.cuda.empty_cache()
            
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           # --- [修改片段 9 修改]: 使用動態命名的 pkl ---
                           os.path.join(self.save_param_dir, self.pkl_name))
                           # -----------------------------------------
            elif mark == 'esc':
                break
            else:
                continue
        # --- [修改片段 10 修改]: 載入時同樣使用動態命名的 pkl ---
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, self.pkl_name)))
        # ----------------------------------------------------
        results0,results1,results2,results3 = self.test(self.test_loader)
        print(results0)
        return results0, os.path.join(self.save_param_dir, 'parameter_clip111.pkl')

    def test(self, dataloader):
        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                batch_category = batch_data['category']
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list= self.model(**batch_data)

                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred0 = final_label_pred_list
                batch_label_pred1 = fusion_label_pred_list
                batch_label_pred2 = image_label_pred_list
                batch_label_pred3 = text_label_pred_list


                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7],label[idxs.squeeze() == 8]))
                batch_category = torch.sort(batch_category).values
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(batch_label_pred0.detach().cpu().numpy().tolist())
                pred1.extend(batch_label_pred1.detach().cpu().numpy().tolist())
                pred2.extend(batch_label_pred2.detach().cpu().numpy().tolist())
                pred3.extend(batch_label_pred3.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metricsTrueFalse(label1, pred0, category, self.category_dict),metricsTrueFalse(label1, pred1, category, self.category_dict),metricsTrueFalse(label1, pred2, category, self.category_dict),metricsTrueFalse(label1, pred3, category, self.category_dict)
