# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

"""
Transformer component:
    Encoder:
        multiple EncoderLayer 

    Decoder:
        multiple Decoder
"""

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """
        
        Parameters
        ----------
        d_model : {int, scalar} dim of the model input and output(default: 512)

        nhead : {int, scalar} parallel attention heads(default: 8)

        num_encoder_layers : {int, scalar} encoder layer number(default: 6)

        num_decoder_layers : {int, scalar} decoder layer number(default: 6)

        dim_feedforward : {int, scalar} FFN layer hidden neurons(default: 2048)

        dropout : {float, scalar} a Dropout layer on attn_output_weights.(default: 0.1)

        activation : {str-like, scalar} ("relu"(default), "gelu", "glu")

        normalize_before : {bool, scalar} False(default), Norm Layer whether before SA or FFN

        return_intermediate_dec : {bool, scalar}
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Parameters
        ----------
        src : {int, tensor(4-dim)} of shape (batch_size, d_model, H, W)

        mask : {boolean, matrix} of shape (batch_size, H, W)

        query_embed : {int, matrix} of shape (num_queries, d_model)

        pos_embed : {int, tensor(4-dim)} of shape (batch_size, d_model, H, W)

        Returns
        -------
        hs : 
                Not aux_loss : {float, tensor(4-dim)} of shape (1, num_queries, batch_size, d_model)
                aux_loss : {float, tensor(4-dim)} of shape (num_decoder_layers, num_queries, batch_size, d_model)

        memory : {float, tensor(4-dim)} of shape (batch_size, d_model, H, W)
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        # src : (batch_size, d_model, H, W) -> (HW, batch_size, d_model)
        src = src.flatten(2).permute(2, 0, 1)
        # pos_embed : (batch_size, d_model, H, W) -> (HW, batch_size, d_model)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed : (num_queries, d_model) -> (num_queries, 1, d_model) -> (num_queries, batch_size, d_model)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask : (batch_size, H, W) -> (batch_size, HW)
        mask = mask.flatten(1)
        # tgt : (num_queries, batch_size, d_model)
        tgt = torch.zeros_like(query_embed)
        # memory : (HW, batch_size, d_model)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # hs : (1, num_queries, batch_size, d_model)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Parameters
        ----------
        encoder_layer : {function-name, scalar} 

        num_layers : {int, scalar} encoder layer number

        norm : {function-name, scalar} norm layer function , (default: None)

        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """

        Parameters
        ----------
        src : {float, tensor(3-dim)} of shape (HW, batch_size, d_model)

        mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size*nheads, HW, HW) . Defaults to None.

        src_key_padding_mask :{Optional[Tensor], matrix} of shape (batch_size, HW). Defaults to None.

        pos : {Optional[Tensor], tensor(3-dim)} of shape (HW, batch_size, d_model)

        Returns
        -------
        output : {float, tensor(3-dim)} of shape (HW, batch_size, d_model)
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        Parameters
        ----------
        decoder_layer : {function-name, scalar} 

        num_layers : {int, scalar} decoder layer number

        norm : {function-name, scalar} norm layer function , (default: None)

        return_intermediate : {boolean, scalar} (default: False)

        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        
        Parameters
        ----------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)

        memory : {float, tensor(3-dim)} of shape (HW, batch_size, d_model), the last encoder layer result

        tgt_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queries, num_queries), default is None.

        memory_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queires, HW), default is None.

        tgt_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, num_queries), default is None.

        memory_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, HW), default is None.

        pos : {Optional[Tensor], tensor(3-dim)} of shape (HW, batch_size, d_model), memory pos. default is None.

        query_pos : {Optional[Tensor], tensor(3-dim)} of shape (num_queries, batch_size, d_model), target query pos. default is None.

        Returns
        -------
        result : 
                Not aux_loss : {float, tensor(4-dim)} of shape (1, num_queries, batch_size, d_model)
                aux_loss : {float, tensor(4-dim)} of shape (num_decoder_layers, num_queries, batch_size, d_model)
        """

        # output : (num_queries, batch_size, d_model)
        output = tgt

        # 将每一层decoder的输出保存
        intermediate = []

        """
        每一层的输出结果都会作为下一层的输入，然后每一个中间层输出是否需要被返回
        """
        for layer in self.layers:
            # output : (num_queries, batch_size, d_model)
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 用于auxiliary decoding losses
        # 把每一层的输出Layer Norm后去预测FFN
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # concatenates a sequence of tensors along a new dimension
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        
        Parameters
        ----------
        d_model : {int, scalar} dim of the model input and output(default: 512)

        nhead : {int, scalar} parallel attention heads(default: 8)

        dim_feedforward : {int, scalar} FFN layer hidden neurons(default: 2048)

        dropout : {float, scalar} a Dropout layer on attn_output_weights.(default: 0.1)

        activation : {str-like, scalar} ("relu"(default), "gelu", "glu")

        normalize_before : {bool, scalar} False(default), Norm Layer whether before SA or FFN
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """forward post, LN after MHA

        Args:
            src (float, tensor(3-dim)): shape of (HW, batch_size, d_model) input data
            src_mask (Optional[Tensor], optional): shape of (batch_size*nheads, HW, HW) . Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): shape of (batch_size, HW). Defaults to None.
            pos (Optional[Tensor], optional): shape of (HW, batch_size, d_model). Defaults to None.

        Returns:
            src: {float, tensor(3-dim)} of shape (HW, batch_size, d_model)
        """
        # -------------------
        # MHA + norm
        #
        # q, k : (HW, batch_size, d_model)
        q = k = self.with_pos_embed(src, pos)

        # nn.MultiheadAttention.forward(q, k, v):
        #       Shapes of intputs:
        #               query : (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        #               key : (S, N, E) where S is the source sequence length, N is the batch size, E is the embedding dimension
        #               value : (S, N, E) where S is the source sequence length, N is the batch size, E is the embedding dimension
        
        # src2 : (HW, batch_size, d_model)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # -------------

        # -------------
        # FFN + norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # -------------

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """forward pre, LN before MHA

        Args:
            src (float, tensor(3-dim)): shape of (HW, batch_size, d_model) input data
            src_mask (Optional[Tensor], optional): shape of (batch_size*nheads, d_model, HW) . Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): shape of (batch_size, HW). Defaults to None.
            pos (Optional[Tensor], optional): shape of (HW, batch_size, d_model). Defaults to None.

        Returns:
            src: {float, tensor(3-dim)} of shape (HW, batch_size, d_model)
        """
        # -------------------
        # Norm + MHA
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        # --------------------

        # --------------------
        # Norm + FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        # --------------------

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """forward post, LN after MHA

        Args:
            src (float, tensor(3-dim)): shape of (HW, batch_size, d_model) input data
            src_mask (Optional[Tensor], optional): shape of (batch_size*nheads, d_model, HW) . Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): shape of (batch_size, HW). Defaults to None.
            pos (Optional[Tensor], optional): shape of (HW, batch_size, d_model). Defaults to None.

        Returns:
            src: {float, tensor(3-dim)} of shape (HW, batch_size, d_model)
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        
        Parameters
        ----------
        d_model : {int, scalar} dim of the model input and output(default: 512)

        nhead : {int, scalar} parallel attention heads(default: 8)

        dim_feedforward : {int, scalar} FFN layer hidden neurons(default: 2048)

        dropout : {float, scalar} a Dropout layer on attn_output_weights. between [0, 1] (default: 0.1)

        activation : {str-like, scalar} ("relu"(default), "gelu", "glu")

        normalize_before : {bool, scalar} False(default), Norm Layer whether before SA or FFN
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """ LN after MHA

        Parameters
        ----------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)

        memory : {float, tensor(3-dim)} of shape (HW, batch_size, d_model), the last encoder layer result

        tgt_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queries, num_queries), default is None.

        memory_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queires, HW), default is None.

        tgt_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, num_queries), default is None.

        memory_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, HW), default is None.
        
        pos : {Optional[Tensor], tensor(3-dim)} of shape (HW, batch_size, d_model), memory pos. default is None.

        query_pos : {Optional[Tensor], tensor(3-dim)} of shape (num_queries, batch_size, d_model), target query pos. default is None.
        
        Returns
        -------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)
        """

        # ---------------
        # 第一层MHA + LN
        #
        #
        # 解码器的输入由输出的embeding(tgt),经过一个多头注意力机制模块后的输出(tgt2 + tgt)
        # 再和编码器的输出(memory)作为下一个attention模块的输入
        #
        # q, k : (num_queries, batch_size, d_model)
        q = k = self.with_pos_embed(tgt, query_pos)

        # tgt2 : (num_queries, batch_size, d_model)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # tgt : (num_queries, batch_size, d_model)
        tgt = self.norm1(tgt)
        # ---------------

        # ---------------
        # 第二层MHA + LN
        #
        # 下面代码对应：
        # (tgt2+tgt)再和编码器的输出(memory)作为下一个attention模块的输入
        #
        # tgt2 : (num_queries, batch_size, d_model)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        # tgt : (num_queries, batch_size, d_model)
        tgt = self.norm2(tgt)
        # ---------------

        # ---------------
        # FFN + Norm
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # ---------------

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """ LN before MHA

        Parameters
        ----------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)

        memory : {float, tensor(3-dim)} of shape (HW, batch_size, d_model), the last encoder layer result

        tgt_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queries, num_queries), default is None.

        memory_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queires, HW), default is None.

        tgt_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, num_queries), default is None.

        memory_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, HW), default is None.
        
        pos : {Optional[Tensor], tensor(3-dim)} of shape (HW, batch_size, d_model), memory pos. default is None.

        query_pos : {Optional[Tensor], tensor(3-dim)} of shape (num_queries, batch_size, d_model), target query pos. default is None.
        
        Returns
        -------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)
        """

        # ------------------
        # 第一层 LN + MHA
        # 
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # ------------------

        # ------------------
        # 第二层 LN + MHA
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        # ------------------

        # ------------------
        # LN + FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        # ------------------

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """

        Parameters
        ----------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)

        memory : {float, tensor(3-dim)} of shape (HW, batch_size, d_model), the last encoder layer result

        tgt_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queries, num_queries), default is None.

        memory_mask : {Optional[Tensor], tensor(3-dim)} of shape (batch_size * nhead, num_queires, HW), default is None.

        tgt_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, num_queries), default is None.

        memory_key_padding_mask : {Optional[Tensor], matrix} of shape (batch_size, HW), default is None.
        
        pos : {Optional[Tensor], tensor(3-dim)} of shape (HW, batch_size, d_model), memory pos. default is None.

        query_pos : {Optional[Tensor], tensor(3-dim)} of shape (num_queries, batch_size, d_model), target query pos. default is None.
        
        Returns
        -------
        tgt : {float, tensor(3-dim)} of shape (num_queries, batch_size, d_model)
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
