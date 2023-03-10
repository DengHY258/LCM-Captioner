# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import functional as F
from pythia.common.registry import registry
from pythia.modules.decoders import LanguageDecoder
import math
import numpy as np

########################################
#DExTraUnit
class new_DExTraUnit(nn.Module):
    def __init__(self,dim_in=512,dim_proj_in=128,dim_out=128,width_multiplier=2.0,dextra_depth=4,dextra_dropout=0.1,max_glt_groups=8,glt_shuffle=True):
        super(new_DExTraUnit,self).__init__()
        self.dim_in = dim_in
        self.dim_proj_in = dim_proj_in
        self.dim_out = dim_out
        self.width_multiplier = width_multiplier
        self.dextra_depth = dextra_depth
        self.dextra_dropout = dextra_dropout
        self.max_glt_groups = max_glt_groups
        self.glt_shuffle = glt_shuffle
        self.max_features = dim_in * self.width_multiplier

        self.obj_input_layer = Linear_layer(dim_in=self.dim_in,dim_out=self.dim_proj_in,dropout=self.dextra_dropout)
        self.ocr_input_layer = Linear_layer(dim_in=self.dim_in,dim_out=self.dim_proj_in,dropout=self.dextra_dropout)

        self.obj_glt_layer1 = Glt_Linear_layer(dim_in=self.dim_proj_in,dim_out=self.dim_proj_in,n_groups=2,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.obj_glt_layer2 = Glt_Linear_layer(dim_in=256,dim_out=1024,n_groups=4,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.obj_dense_layer = Linear_layer(dim_in=1024,dim_out=512,dropout=self.dextra_dropout)

        self.ocr_glt_layer1 = Glt_Linear_layer(dim_in=self.dim_proj_in,dim_out=self.dim_proj_in,n_groups=2,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.ocr_glt_layer2 = Glt_Linear_layer(dim_in=256,dim_out=1024,n_groups=4,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.ocr_dense_layer = Linear_layer(dim_in=1024,dim_out=512,dropout=self.dextra_dropout)


        self.shared_layer = Glt_Linear_layer(dim_in=1024,dim_out=512,n_groups=8,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        
        self.obj_glt_layer3 = Glt_Linear_layer(dim_in=1152,dim_out=1024,n_groups=4,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.obj_glt_layer4 = Glt_Linear_layer(dim_in=1152,dim_out=128,n_groups=2,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.obj_output_layer = Linear_layer(dim_in=self.dim_out + self.dim_proj_in,dim_out=self.dim_out,dropout=self.dextra_dropout)

        self.ocr_glt_layer3 = Glt_Linear_layer(dim_in=1152,dim_out=1024,n_groups=4,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.ocr_glt_layer4 = Glt_Linear_layer(dim_in=1152,dim_out=128,n_groups=2,dropout=self.dextra_dropout,use_shuffle=self.glt_shuffle)
        self.ocr_output_layer = Linear_layer(dim_in=self.dim_out + self.dim_proj_in,dim_out=self.dim_out,dropout=self.dextra_dropout)

    def forward(self,obj_feature,ocr_feature):
        obj_feature = self.obj_input_layer(obj_feature)
        ocr_feature = self.ocr_input_layer(ocr_feature)

        n_dims = obj_feature.dim()
        if n_dims == 2:
            # [B x N] --> [B x 1 x N]
            obj_feature = obj_feature.unsqueeze(dim=1)
            ocr_feature = ocr_feature.unsqueeze(dim=1)  # add dummy T dimension
            # [B x 1 x N] --> [B x 1 x M]
            obj_feature,ocr_feature = self.forward_dextra(obj_feature,ocr_feature)
            # [B x 1 x M] --> [B x M]
            obj_feature = obj_feature.squeeze(dim=1)  # remove dummy T dimension
            ocr_feature = ocr_feature.squeeze(dim=1)
        elif n_dims == 3:
            obj_feature,ocr_feature = self.forward_dextra(obj_feature,ocr_feature)
        else:
            raise NotImplementedError
        return obj_feature,ocr_feature

    def combine(self,out,x,g_next):
        B = x.size(0)
        T = x.size(1)
        x_g = x.contiguous().view(B, T, g_next, -1)

        out = out.contiguous().view(B, T, g_next, -1)

        out = torch.cat([x_g, out], dim=-1)

        out = out.contiguous().view(B, T, -1)

        return out

    def share_combine(self,share_out,temp_out,x,g_next):
        B = x.size(0)
        T = x.size(1)
        x_g = x.contiguous().view(B, T, g_next, -1)

        share_out = share_out.contiguous().view(B, T, g_next, -1)
        temp_out = temp_out.contiguous().view(B, T, g_next, -1)

        out = torch.cat([x_g, share_out,temp_out], dim=-1)

        out = out.contiguous().view(B, T, -1)

        return out

    def forward_dextra(self,obj_feature,ocr_feature):
        B = obj_feature.size(0)
        obj_T = obj_feature.size(1)
        ocr_T = ocr_feature.size(1)

        obj_out = obj_feature
        ocr_out = ocr_feature

        obj_out = self.obj_glt_layer1(obj_out)
        ocr_out = self.ocr_glt_layer1(ocr_out)

        # obj_feature_g = obj_feature.contiguous().view(B, obj_T, 4, -1)
        # obj_out = obj_out.contiguous().view(B, obj_T, 4, -1)
        # obj_out = torch.cat([obj_feature_g, obj_out], dim=-1)
        # obj_out = obj_out.contiguous().view(B, obj_T, -1)
        obj_out = self.combine(obj_out,obj_feature,4)
        ocr_out = self.combine(ocr_out,ocr_feature,4)

        obj_out = self.obj_glt_layer2(obj_out)
        ocr_out = self.obj_glt_layer2(ocr_out)


        shared_obj_out = self.shared_layer(obj_out)
        shared_ocr_out = self.shared_layer(ocr_out)

        temp_obj_out = self.obj_dense_layer(obj_out)
        temp_ocr_out = self.obj_dense_layer(ocr_out)

        obj_out = self.share_combine(shared_obj_out,temp_obj_out,obj_feature,4)
        ocr_out = self.share_combine(shared_ocr_out,temp_ocr_out,ocr_feature,4)

        obj_out = self.obj_glt_layer3(obj_out)
        ocr_out = self.ocr_glt_layer3(ocr_out)

        obj_out = self.combine(obj_out,obj_feature,2)
        ocr_out = self.combine(ocr_out,ocr_feature,2)

        obj_out = self.obj_glt_layer4(obj_out)
        ocr_out = self.ocr_glt_layer4(ocr_out)

        obj_out = torch.cat([obj_feature, obj_out], dim=-1)
        ocr_out = torch.cat([ocr_feature, ocr_out], dim=-1)

        obj_out = self.obj_output_layer(obj_out)
        ocr_out = self.ocr_output_layer(ocr_out)

        return obj_out,ocr_out


class Linear_layer(nn.Module):
    def __init__(self,dim_in,dim_out,num_gates=1,dropout=0.0,use_act_fn=False):
        super(Linear_layer,self).__init__()
        self.use_act_fn =use_act_fn
        self.weights = torch.nn.Parameter(torch.Tensor(dim_out * num_gates, dim_in))
        self.bias = torch.nn.Parameter(torch.Tensor(dim_out * num_gates))
        self.normalization_fn = BatchNorm(num_features=dim_out*num_gates)
        self.dropout_p = dropout
        if self.dropout_p == 0.0:
            self.use_dropout = False
        else:
            self.use_dropout = True
        if self.use_dropout:
            self.drop_layer = nn.Dropout(p=dropout)
        if self.use_act_fn:
            self.act_fn = GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        nn.init.constant_(self.bias.data, 0)

    def forward(self,x):
        x = F.linear(x, weight=self.weights, bias=self.bias)
        x = self.normalization_fn(x)
        if self.use_act_fn:
            x = self.act_fn(x)
        if self.use_dropout:
            x = self.drop_layer(x)
        return x

class Glt_Linear_layer(nn.Module):
    def __init__(self,dim_in,dim_out,n_groups,dropout=0.01,use_shuffle=False):
        super(Glt_Linear_layer,self).__init__()
        if dim_in % n_groups !=0:
            print('in_groups wrong')
        if dim_out % n_groups !=0:
            print('out_groups wrong')
        in_groups = dim_in // n_groups
        out_groups = dim_out // n_groups
        
        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        self.normalization_fn = BatchNorm(num_features=out_groups)
        self.drop_layer = nn.Dropout(p=dropout)
        self.act_fn = GELU()
        self.use_shuffle = use_shuffle
        self.n_groups = n_groups
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        nn.init.constant_(self.bias.data, 0)
    def process_input_bmm(self, x):
        #N --> Input dimension,M --> Output dimension,g --> groups,G --> gates,param x: Input of dimension B x N,return: Output of dimension B x M
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights
        x = torch.add(x, self.bias)
        if self.use_shuffle:
            # [g x B x M/g] --> [B x M/g x g]
            x = x.permute(1, 2, 0)
            # [B x M/g x g] --> [B x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B x M/g] --> [B x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first
        x = self.normalization_fn(x)
        x = self.act_fn(x)
        return x
    def forward(self, x):
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError
        x = self.drop_layer(x)
        return x

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, affine=affine)

    def forward(self, x):
        if x.dim() == 3:
            bsz, seq_len, feature_size = x.size()
            out = self.layer(x.view(-1, feature_size))
            return out.contiguous().view(bsz, seq_len, -1)
        else:
            return self.layer(x)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return nn.functional.gelu(x)

class SingleHeadAttention(nn.Module):
    def __init__(self,q_in_dim, kv_in_dim, proj_dim, out_dim,dropout=0.01,cross_attention = True):
        super(SingleHeadAttention,self).__init__()
        self.q_embed_dim = q_in_dim
        self.kv_embed_dim = kv_in_dim
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.cross_attention = cross_attention

        if self.cross_attention:
            self.linear_q = Linear_layer(dim_in=self.q_embed_dim,dim_out=self.proj_dim,num_gates=1)
            self.linear_kv = Linear_layer(dim_in=self.kv_embed_dim,dim_out=self.proj_dim,num_gates=2)
        else:
            assert q_in_dim == kv_in_dim
            self.linear_kqv = Linear_layer(dim_in=self.q_embed_dim,dim_out=self.proj_dim,num_gates=3)

        self.scaling = self.proj_dim ** -0.5
        self.output_layer = Linear_layer(dim_in=self.proj_dim,dim_out=self.out_dim)
    def forward(self,query,key = None,attn_mask = None):
        #sql_len, b_s, q_embed_dim = query.size()
        b_s, sql_len,q_embed_dim = query.size()

        if self.cross_attention:
            q = self.linear_q(query)
            k, v = torch.chunk(self.linear_kv(key), chunks=2, dim=-1)
        else:
            q, k, v = torch.chunk(self.linear_kqv(query), chunks=3, dim=-1)
        
        #q = q * self.scaling
        # print(q.shape)
        # print(k.shape)
        # q = q.contiguous().transpose(0, 1)
        # k = k.contiguous().transpose(0, 1)
        # v = v.contiguous().transpose(0, 1)
        
        src_len = k.size(1)
        # [B x T x C] x [B x C x S] --> [B x T x S]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        #print(attn_weights.shape)
        attn_weights = attn_weights /math.sqrt(self.proj_dim)

        assert list(attn_weights.size()) == [b_s, sql_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training
        )
        # [B x T x S] x [B x S x F] --> [B x T x F]
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [b_s, sql_len, self.proj_dim]
        # [B x T x F] --> [T x B x F]
        #attn = attn.transpose(0, 1).contiguous()

        # [T x B x F] --> [ T x B x F']
        attn = self.output_layer(attn)

        return attn

class Light_FFN(nn.Module):
    def __init__(self,dim_in,dim_out=768,ffn_red_factor=4,ffn_dropout=0.1):
        super(Light_FFN,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout_p = ffn_dropout
        self.light_ffn_dim = self.dim_in // ffn_red_factor
        self.fc1 = Linear_layer(dim_in=self.dim_in,dim_out=self.light_ffn_dim)
        self.fc2 = Linear_layer(dim_in=self.light_ffn_dim,dim_out=self.dim_in)
        self.fc3 = nn.Linear(self.dim_in,self.dim_out)
        self.final_layer_norm = BatchNorm(num_features=self.dim_out)
        self.act_fn = GELU()

    def forward(self,x):
        residual = x
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = residual + x
        x = self.fc3(x)
        x = self.final_layer_norm(x)
        return x

class CrossAttnDeLightModule(nn.Module):
    def __init__(self,dropout_p=0.1):
        super(CrossAttnDeLightModule,self).__init__()
        self.dextra_layer = new_DExTraUnit()
        self.obj_attn_layer = SingleHeadAttention(q_in_dim=128,kv_in_dim=128,proj_dim=128,out_dim=512,cross_attention=True)
        self.obj_layer_norm = BatchNorm(num_features=512)
        self.obj_light_FFN = Light_FFN(dim_in=512)

        #self.ocr_dextra_layer = DExTraUnit()
        self.ocr_attn_layer = SingleHeadAttention(q_in_dim=128,kv_in_dim=128,proj_dim=128,out_dim=512,cross_attention=True)
        self.ocr_layer_norm = BatchNorm(num_features=512)
        self.ocr_light_FFN = Light_FFN(dim_in=512)

        self.dropout = dropout_p
        
    def forward(self,obj_feature,ocr_feature):
        obj_residual = obj_feature
        ocr_residual = ocr_feature

        # obj_feature = self.obj_dextra_layer(obj_feature)
        # ocr_feature = self.obj_dextra_layer(ocr_feature)
        new_obj_feature,new_ocr_feature = self.dextra_layer(obj_feature,ocr_feature)

        obj_attn = self.obj_attn_layer(new_obj_feature,new_ocr_feature)
        ocr_attn = self.ocr_attn_layer(new_ocr_feature,new_obj_feature)

        obj_attn = F.dropout(obj_attn, p=self.dropout, training=self.training)
        ocr_attn = F.dropout(ocr_attn, p=self.dropout, training=self.training)

        new_obj_feature = obj_residual + obj_attn
        new_ocr_feature = ocr_residual + ocr_attn

        new_obj_feature = self.obj_layer_norm(new_obj_feature)
        new_ocr_feature = self.ocr_layer_norm(new_ocr_feature)

        obj_output = self.obj_light_FFN(new_obj_feature)
        ocr_output = self.obj_light_FFN(new_ocr_feature)

        return obj_output,ocr_output


########################################
class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding_size="same",
        pool_stride=2,
        batch_norm=True,
    ):
        super().__init__()

        if padding_size == "same":
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norm_2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))

        if self.batch_norm:
            x = self.batch_norm_2d(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        if input.dim() > 1:
            input = input.view(input.size(0), -1)

        return input

class UnFlatten(nn.Module):
    def forward(self, input, sizes=[]):
        return input.view(input.size(0), *sizes)


class GatedTanh(nn.Module):
    """
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\in R^m => y \in R^n
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))

        # Element wise multiplication
        y = y_tilda * gated

        return y


# TODO: Do clean implementation without Sequential
class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ClassifierLayer(nn.Module):
    def __init__(self, classifier_type, in_dim, out_dim, **kwargs):
        super(ClassifierLayer, self).__init__()

        if classifier_type == "weight_norm":
            self.module = WeightNormClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == "logit":
            self.module = LogitClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == "language_decoder":
            self.module = LanguageDecoder(in_dim, out_dim, **kwargs)
        elif classifier_type == "linear":
            self.module = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError("Unknown classifier type: %s" % classifier_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LogitClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(LogitClassifier, self).__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        text_non_linear_dim = kwargs["text_hidden_dim"]
        image_non_linear_dim = kwargs["img_hidden_dim"]

        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)

        if "pretrained_image" in kwargs and kwargs["pretrained_text"] is not None:
            self.linear_text.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_text"])
            )

        if "pretrained_image" in kwargs and kwargs["pretrained_image"] is not None:
            self.linear_image.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_image"])
            )

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super(WeightNormClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModalCombineLayer(nn.Module):
    def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
        super(ModalCombineLayer, self).__init__()
        if combine_type == "MFH":
            self.module = MFH(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "non_linear_element_multiply":
            self.module = NonLinearElementMultiply(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "two_layer_element_multiply":
            self.module = TwoLayerElementMultiply(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "top_down_attention_lstm":
            self.module = TopDownAttentionLSTM(img_feat_dim, txt_emb_dim, **kwargs)
        else:
            raise NotImplementedError("Not implemented combine type: %s" % combine_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MfbExpand(nn.Module):
    def __init__(self, img_feat_dim, txt_emb_dim, hidden_dim, dropout):
        super(MfbExpand, self).__init__()
        self.lc_image = nn.Linear(in_features=img_feat_dim, out_features=hidden_dim)
        self.lc_ques = nn.Linear(in_features=txt_emb_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = torch.unsqueeze(ques1, 1).expand(-1, num_location, -1)
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class MFH(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(MFH, self).__init__()
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.relu = nn.ReLU()

        hidden_sizes = kwargs["hidden_sizes"]
        self.out_dim = int(sum(hidden_sizes) / kwargs["pool_size"])

        self.order = kwargs["order"]
        self.pool_size = kwargs["pool_size"]

        for i in range(self.order):
            mfb_exp_i = MfbExpand(
                img_feat_dim=image_feat_dim,
                txt_emb_dim=ques_emb_dim,
                hidden_dim=hidden_sizes[i],
                dropout=kwargs["dropout"],
            )
            self.mfb_expand_list.append(mfb_exp_i)
            self.mfb_sqz_list.append(self.mfb_squeeze)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1

        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)

        # append at last feature
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature

    def mfb_squeeze(self, joint_feature):
        # joint_feature dim: N x k x dim or N x dim

        orig_feature_size = len(joint_feature.size())

        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)

        batch_size, num_loc, dim = joint_feature.size()

        if dim % self.pool_size != 0:
            exit(
                "the dim %d is not multiply of \
             pool_size %d"
                % (dim, self.pool_size)
            )

        joint_feature_reshape = joint_feature.view(
            batch_size, num_loc, int(dim / self.pool_size), self.pool_size
        )

        # N x 100 x 1000 x 1
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)

        iatt_iq_sqrt = torch.sqrt(self.relu(iatt_iq_sumpool)) - torch.sqrt(
            self.relu(-iatt_iq_sumpool)
        )

        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
        iatt_iq_l2 = nn.functional.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.pool_size))

        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

        return iatt_iq_l2


# need to handle two situations,
# first: image (N, K, i_dim), question (N, q_dim);
# second: image (N, i_dim), question (N, q_dim);
class NonLinearElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(NonLinearElementMultiply, self).__init__()
        self.fa_image = ReLUWithWeightNormFC(image_feat_dim, kwargs["hidden_dim"])
        self.fa_txt = ReLUWithWeightNormFC(ques_emb_dim, kwargs["hidden_dim"])

        context_dim = kwargs.get("context_dim", None)
        if context_dim is not None:
            self.fa_context = ReLUWithWeightNormFC(context_dim, kwargs["hidden_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding, context_embedding=None):
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)

        if len(image_feat.size()) == 3:
            question_fa_expand = question_fa.unsqueeze(1)
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand

        if context_embedding is not None:
            context_fa = self.fa_context(context_embedding)

            context_text_joint_feaure = context_fa * question_fa_expand
            joint_feature = torch.cat([joint_feature, context_text_joint_feaure], dim=1)

        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TopDownAttentionLSTM(nn.Module):
    def __init__(self, image_feat_dim, embed_dim, **kwargs):
        super().__init__()
        self.fa_image = weight_norm(nn.Linear(image_feat_dim, kwargs["attention_dim"]))
        self.fa_hidden = weight_norm(
            nn.Linear(kwargs["hidden_dim"], kwargs["attention_dim"])
        )
        self.top_down_lstm = nn.LSTMCell(
            embed_dim + image_feat_dim + kwargs["hidden_dim"],
            kwargs["hidden_dim"],
            bias=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["attention_dim"]

    def forward(self, image_feat, embedding):
        image_feat_mean = image_feat.mean(1)

        # Get LSTM state
        state = registry.get("{}_lstm_state".format(image_feat.device))
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        h1, c1 = self.top_down_lstm(
            torch.cat([h2, image_feat_mean, embedding], dim=1), (h1, c1)
        )

        state["td_hidden"] = (h1, c1)

        image_fa = self.fa_image(image_feat)
        hidden_fa = self.fa_hidden(h1)

        joint_feature = self.relu(image_fa + hidden_fa.unsqueeze(1))
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TwoLayerElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(TwoLayerElementMultiply, self).__init__()

        self.fa_image1 = ReLUWithWeightNormFC(image_feat_dim, kwargs["hidden_dim"])
        self.fa_image2 = ReLUWithWeightNormFC(
            kwargs["hidden_dim"], kwargs["hidden_dim"]
        )
        self.fa_txt1 = ReLUWithWeightNormFC(ques_emb_dim, kwargs["hidden_dim"])
        self.fa_txt2 = ReLUWithWeightNormFC(kwargs["hidden_dim"], kwargs["hidden_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding):
        image_fa = self.fa_image2(self.fa_image1(image_feat))
        question_fa = self.fa_txt2(self.fa_txt1(question_embedding))

        if len(image_feat.size()) == 3:
            num_location = image_feat.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
                -1, num_location, -1
            )
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TransformLayer(nn.Module):
    def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
        super(TransformLayer, self).__init__()

        if transform_type == "linear":
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == "conv":
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError(
                "Unknown post combine transform type: %s" % transform_type
            )
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=hidden_dim, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1
        )
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

        return iatt_conv3


class BCNet(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act="ReLU", dropout=[0.2, 0.5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])

        if k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out is None:
            pass

        elif h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * self.k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if self.h_out is None:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = d_.transpose(1, 2).transpose(2, 3)
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))
            logits = logits + self.h_bias
            return logits

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            return logits.transpose(2, 3).transpose(1, 2)

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)
        logits = logits.squeeze(3).squeeze(2)

        if self.k > 1:
            logits = logits.unsqueeze(1)
            logits = self.p_net(logits).squeeze(1) * self.k

        return logits


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

            if act is not None:
                layers.append(getattr(nn, act)())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))

        if act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[0.2, 0.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name="h_mat",
            dim=None,
        )

    def forward(self, v, q, v_mask=True):
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)

        if v_mask:
            v_abs_sum = v.abs().sum(2)
            mask = (v_abs_sum == 0).unsqueeze(1).unsqueeze(3)
            mask = mask.expand(logits.size())
            logits.masked_fill_(mask, -float("inf"))

        expanded_logits = logits.view(-1, self.glimpse, v_num * q_num)
        p = nn.functional.softmax(expanded_logits, 2)

        return p.view(-1, self.glimpse, v_num, q_num), logits
