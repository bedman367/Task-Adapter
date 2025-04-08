import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import math
import clip
from einops import rearrange

from module_adapter import clip_vit_base_patch16_adapter
from utils import *

class MetaTemplate(nn.Module):
    def __init__(self, n_way, n_support, n_query):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
    
    @abstractmethod
    def forward(self,x):
        pass
    
    def distribute_backbone(self,devices):
        self.feature.cuda(0)
        self.feature = torch.nn.DataParallel(self.feature,device_ids = devices)
    


class ProtoNet(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, dataset):
        super().__init__(n_way, n_support, n_query)


        
        model,prepocess = clip.load("RN50",device="cuda",jit=False)#ViT-B/16
        model.float()
        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()

        with torch.no_grad():
            self.sem_embedding = model.encode_text(self.cls_txt).to(torch.float32)

        self.feature = model.visual

        # partial fine-tuning
        for name, p in self.feature.named_parameters():
            if 'attnpool' not in name :
                p.requires_grad_(False)
        # for name, p in self.feature.named_parameters():
        #     if 'transformer.resblocks.11' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
        #         p.requires_grad_(False)
        

    
    def forward(self, x, label):

        label_idx = list(label[:,0].numpy())
        sem_feat = self.sem_embedding[label_idx]# 5 * sem_dim
     
        backbone_feat = self.feature(x)
        backbone_feat = backbone_feat.reshape(self.n_way, (self.n_support+self.n_query), 8, -1) # 5 x 2 x 8 x 2048
        z_support   = backbone_feat[:, :self.n_support].reshape(self.n_way*self.n_support, 8, -1)
        z_query     = backbone_feat[:, self.n_support:].reshape(self.n_way*self.n_query, 8, -1)

       

        z_proto = z_support.reshape(self.n_way, self.n_support, 8, -1).mean(1)
        z_query = z_query.reshape(self.n_way*self.n_query, 8, -1 )


        sem_dists = cosine_similarity(z_query.mean(1), sem_feat)

        dists = cosine_similarity(z_query.mean(1), z_proto.mean(1)).squeeze()




        return dists, sem_dists
    


class TaskAdapter(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, dataset):
        super().__init__(n_way, n_support, n_query)


        
        model,prepocess = clip.load("ViT-B/16",device="cuda",jit=False)
        model.float()
        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()

        with torch.no_grad():
            self.sem_embedding = model.encode_text(self.cls_txt).to(torch.float32)

        self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers=6)


        
    
    def forward(self, x, label):

        label_idx = list(label[:,0].numpy())
        sem_feat = self.sem_embedding[label_idx]# 5 * sem_dim

        # normal flow

        N,C,H,W = x.shape
        x = x.reshape(self.n_way*(self.n_query+self.n_support), 8 , C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # B C T H W

        x = x.reshape(self.n_way, (self.n_query+self.n_support), C , 8, H, W)
    
        support_images = x[:, :self.n_support].reshape(self.n_way*self.n_support, C, 8, H, W)
        target_images = x[:,self.n_support:].reshape(self.n_way*self.n_query, C, 8, H, W)


        z_support  = self.feature(support_images).reshape(self.n_way*self.n_support, 8, -1)
        z_query = self.feature(target_images).reshape(self.n_way*self.n_query, 8, -1)

        z_proto = z_support.reshape(self.n_way, self.n_support, 8, -1).mean(1)
        z_query = z_query.reshape(self.n_way*self.n_query, 8, -1 )


        sem_dists = cosine_similarity(z_query.mean(1), sem_feat)

        dists = cosine_similarity(z_query.mean(1), z_proto.mean(1)).squeeze()




        return dists, sem_dists
    
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x
    

class TemporalCrossTransformer(nn.Module):
    def __init__(self, h_dim,n_way, n_shot, n_query, temp_set) -> None:
        super().__init__()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.temporal_set_size = temp_set
        self.trans_linear_in_dim = h_dim 
        self.trans_linear_out_dim = 1152
        self.seq_len = 8
        self.pe = PositionalEncoding(self.trans_linear_in_dim, dropout=0.1, max_len=int(self.seq_len*1.5))
        self.k_linear = nn.Linear(self.trans_linear_in_dim * self.temporal_set_size, self.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.trans_linear_in_dim * self.temporal_set_size, self.trans_linear_out_dim)
        self.norm_k = nn.LayerNorm(self.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.trans_linear_out_dim)
        self.class_softmax = torch.nn.Softmax(dim=1)

        frame_idx = [i for i in range(self.seq_len)]
        from itertools import combinations 
        frame_combinations = combinations(frame_idx, self.temporal_set_size)

        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples)

    def forward(self, support_set, queries):
        
        n_queries = self.n_way*self.n_query  #if self.training else self.n_way
        n_support = self.n_way*self.n_shot
        assert  n_queries == queries.shape[0]
        assert  n_support == support_set.shape[0]

        support_set = self.pe(support_set)
        queries = self.pe(queries)

        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]# 多帧的特征被拼到一起，2048*temporal_size
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        #只对key做归一化
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        support_labels = torch.from_numpy(np.repeat(range(self.n_way),self.n_shot)).cuda()

        unique_labels = torch.unique(support_labels)

        all_distances_tensor = torch.zeros(n_queries, self.n_way).cuda()

        for label_idx, c in enumerate(unique_labels):

            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1))/math.sqrt(self.trans_linear_out_dim)

            class_scores = class_scores.permute(0,2,1,3).reshape(n_queries, self.tuples_len, -1)# 最后一维是所有类别全部帧元组
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores).reshape(n_queries, self.tuples_len, -1, self.tuples_len).permute(0,2,1,3)

            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim =1)
            
            sim_matrx = cos_sim(mh_queries_vs, query_prototype)
            sim_matrx = torch.mean(sim_matrx,dim=[-2,-1])
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = sim_matrx
        

        return all_distances_tensor

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class TRX(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, temp_set, dataset):
        super().__init__(n_way, n_support, n_query)
        
        self.alpha = torch.nn.Parameter(torch.tensor(0.5),requires_grad=True)
        model, preprocess = clip.load("ViT-B/16",device='cuda',jit=False)#clip.load("ViT-L/14",device='cuda',jit=False)
        model.float()

        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()

        with torch.no_grad():
            self.sem_embedding = model.encode_text(self.cls_txt).to(torch.float32)

        # self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers=6) # task adapter
        self.feature = model.visual
        # partial fine-tuning
        # for name, p in self.feature.named_parameters():
        #     if 'transformer.resblocks.11' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
        #         p.requires_grad_(False)

        self.mid_dim = self.sem_embedding.shape[-1]
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.temporal_atte_before = Transformer_v2(dim = self.mid_dim, heads=8, dim_head_k=self.mid_dim//8, dropout_atte=0.2)
        self.relu = nn.ReLU(inplace=True)
        
        self.transformers = nn.ModuleList([TemporalCrossTransformer(self.mid_dim, n_way, n_support, n_query, s) for s in temp_set])


    def euclidean_dist( x, y, normalize=False):
        # x: N x D
        # y: M x D
        if normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
            y = torch.nn.functional.normalize(y, p=2, dim=-1)
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        # return x@y.T

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


    def forward(self, x, label):

        label_idx = list(label[:,0].numpy().astype(int))
        sem_feat = self.sem_embedding[label_idx] # 5 * sem_dim

        

        backbone_feat = self.feature(x) # 80 x 2048
        backbone_feat = backbone_feat.reshape(self.n_way, self.n_support+self.n_query, 8, -1) # 5 x 2 x 8 x 2048
        z_support   = backbone_feat[:, :self.n_support].reshape(-1, 8, self.mid_dim)
        z_query     = backbone_feat[:, self.n_support:].reshape(-1, 8, self.mid_dim)


        sem_logits = cosine_similarity(z_query.mean(1), sem_feat)

        all_logits = [t(z_support, z_query) for  t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)

        visual_logits = torch.mean(all_logits, dim = -1)#.softmax(-1)

        
        return visual_logits,sem_logits
    


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)


class BiMHM(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, dataset):
        super().__init__(n_way, n_support, n_query)
        self.scale = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.scale.data.fill_(1.0)
        self.relu = nn.ReLU(inplace=True)
        self.mid_dim = 1024

        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.temporal_atte_before = Transformer_v2(dim = self.mid_dim, heads=8, dim_head_k=self.mid_dim//8, dropout_atte=0.2)
        self.classification_layer = nn.Linear(self.mid_dim, 64)

        # self.positional_embedding = nn.Parameter( torch.randn(9, 1024))
        self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)


        model, preprocess = clip.load("ViT-B/16",device='cuda',jit=False)
        model.float()
        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()
       
        with torch.no_grad():
            self.sem_embedding = model.encode_text(self.cls_txt).to(torch.float32)

        # self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers = 6) # task adapter
        self.feature = model.visual
        # partial fine-tuning
        # for name, p in self.feature.named_parameters():
        #     if 'transformer.resblocks.11' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
        #         p.requires_grad_(False)
        
        self.mid_dim = self.sem_embedding.size(-1)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.temporal_atte_before = Transformer_v2(dim = self.mid_dim, heads=8, dim_head_k=self.mid_dim//8, dropout_atte=0.2)
        self.positional_embedding = nn.Parameter( torch.randn(9, 2048))
        self.pe = PositionalEncoding(d_model=self.mid_dim)


    
    def forward(self, x, label):
        # 使用加adapter的backbone
        # N,C,H,W = x.shape
        # x = x.reshape(self.n_way*(self.n_query+self.n_support), 8 , C, H, W)
        # x = x.permute(0, 2, 1, 3, 4) # B C T H W
        # x = x.reshape(self.n_way, (self.n_query+self.n_support), C , 8, H, W)
        # support_images = x[:, :self.n_support].reshape(self.n_way*self.n_support, C, 8, H, W)
        # target_images = x[:,self.n_support:].reshape(self.n_way*self.n_query, C, 8, H, W)
        # z_support  = self.feature(support_images).reshape(self.n_way*self.n_support, 8, -1)
        # z_query = self.feature(target_images).reshape(self.n_way*self.n_query, 8, -1)



        backbone_feat = self.feature(x)
        
        backbone_feat = backbone_feat.reshape(self.n_way, self.n_query+self.n_support, 8 , -1)


        support_features = backbone_feat[:, :self.n_support].reshape(self.n_way*self.n_support, 8, -1)
        target_features = backbone_feat[:,self.n_support:].reshape(self.n_way*self.n_query, 8, -1)        


        label_idx = list(label[:,0].numpy())
        sem_feat = self.sem_embedding[label_idx]# 5 * sem_dim
        class_dists2 = cosine_similarity(target_features.mean(1),sem_feat)
        

        support_features = rearrange(support_features, 'b s d -> (b s) d') # way*shot 8 2048
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)
        frame_dists = frame_sim
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = self.n_way*self.n_query, sb = self.n_way*self.n_support)
    
        cum_dists = 0.5*dists.max(3)[0].sum(2)+0.5*dists.max(2)[0].sum(2)
        dists = cum_dists.reshape(self.n_way*self.n_query, self.n_way, self.n_support).mean(-1)

        return dists, class_dists2, class_dists2, class_dists2
    





def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


class OTAM(MetaTemplate):#最原始版本
    def __init__(self, model_func, n_way, n_support, n_query, dataset):
        super().__init__(model_func, n_way, n_support, n_query)
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query #(change depends on input) 
        model, preprocess = clip.load("ViT-B/16",device='cuda',jit=False)
        model.float()

        self.logit_scale = model.logit_scale
       
        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()                              
                                 

        with torch.no_grad():
            self.sem_embedding = model.encode_text(self.cls_txt).to(torch.float32)#.to(torch.float32)

        # self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers = 6)
        self.feature = model.visual
    
        

    
    def forward(self, x, label):

        # N,C,H,W = x.shape
        # x = x.reshape(self.n_way*(self.n_query+self.n_support), 8 , C, H, W)
        # x = x.permute(0, 2, 1, 3, 4) # B C T H W

        # x = x.reshape(self.n_way, (self.n_query+self.n_support), C , 8, H, W)
        # support_images = x[:, :self.n_support].reshape(self.n_way*self.n_support, C, 8, H, W)
        # target_images = x[:,self.n_support:].reshape(self.n_way*self.n_query, C, 8, H, W)
        # z_support  = self.feature(support_images).reshape(self.n_way*self.n_support, 8, -1)
        # z_query = self.feature(target_images).reshape(self.n_way*self.n_query, 8, -1)
        
        backbone_feat = self.feature(x)
        backbone_feat = backbone_feat.reshape(self.n_way, (self.n_support+self.n_query), 8, -1) # 5 x 2 x 8 x 2048
        z_support   = backbone_feat[:, :self.n_support].reshape(self.n_way*self.n_support, 8, -1)
        z_query     = backbone_feat[:, self.n_support:].reshape(self.n_way*self.n_query, 8, -1)

        label_idx = list(label[:,0].numpy())
        sem_feat = self.sem_embedding[label_idx] # 5 * sem_dim


        scores_2 = cosine_similarity(z_query.mean(1), sem_feat) # q 8 1024 @ 5 1024

        from einops import rearrange
        z_support = rearrange(z_support, 'b s d -> (b s) d')
        z_query = rearrange(z_query, 'b s d -> (b s) d')
        frame_sim = 1 - cos_sim(z_query, z_support)
        dists = rearrange(frame_sim, '(tb ts) (sb ss) -> tb sb ts ss', tb = self.n_way*self.n_query, sb = self.n_way*self.n_support)
        cum_dists = OTAM_cum_dist(dists)+ OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts')).reshape(self.n_way*self.n_query,self.n_way*self.n_support)
        scores = cum_dists.reshape(self.n_way*self.n_query, self.n_way, self.n_support).mean(-1)

        return -scores, scores_2, scores_2, scores_2