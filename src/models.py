import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return mask * inputs + (-1e30) * (1 - mask)


class Initialized_Conv1d(nn.Module):
    """
        Just an initalized convolution module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        relu=False,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, device, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length, channels = x.size()[1], x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(device)).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale * 1.0 / min_timescale) / (
        num_timescales - 1
    )

    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=k,
            groups=in_ch,
            padding=k // 2,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias
        )

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mem_conv = Initialized_Conv1d(
            config.enc_filters,
            config.enc_filters * 2,
            kernel_size=1,
            relu=False,
            bias=False,
        )
        self.query_conv = Initialized_Conv1d(
            config.enc_filters,
            config.enc_filters,
            kernel_size=1,
            relu=False,
            bias=False,
        )

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.config.attention_heads)
        K, V = [
            self.split_last_dim(tensor, self.config.attention_heads)
            for tensor in torch.split(memory, self.config.enc_filters, dim=2)
        ]

        key_depth_per_head = self.config.enc_filters // self.config.attention_heads
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)

        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.config.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [B, L, m]
        n: an integer.
        Returns:
        a Tensor with shape [B, n, L, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, config):
        super().__init__()
        self.config = config

        self.convs = nn.ModuleList(
            [DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)]
        )

        self.self_att = SelfAttention(config)

        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)

        self.norm_C = nn.ModuleList(
            [nn.LayerNorm(config.enc_filters) for _ in range(conv_num)]
        )
        self.norm_1 = nn.LayerNorm(config.enc_filters)
        self.norm_2 = nn.LayerNorm(config.enc_filters)

        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        out = PosEncoder(x, self.config.device)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            # Note: not needed as we are already using layer_dropout here
            # if (i) % 2 == 0:
            #     out = F.dropout(out, p=self.config.dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(
                out, res, self.config.dropout * float(l) / total_layers
            )
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.config.dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(
            out, res, self.config.dropout * float(l) / total_layers
        )
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.config.dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(
            out, res, self.config.dropout * float(l) / total_layers
        )
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        w4C = torch.empty(config.enc_filters, 1)
        w4Q = torch.empty(config.enc_filters, 1)
        w4mlu = torch.empty(1, 1, config.enc_filters)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]

        S = self.trilinear_for_attention(C, Q)

        Cmask = Cmask.view(batch_size_c, self.config.para_limit, 1)
        Qmask = Qmask.view(batch_size_c, 1, self.config.ques_limit)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)

        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.config.dropout, training=self.training)
        Q = F.dropout(Q, p=self.config.dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, self.config.ques_limit])
        subres1 = (
            torch.matmul(Q, self.w4Q)
            .transpose(1, 2)
            .expand([-1, self.config.para_limit, -1])
        )
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res
        # shape of res will be B X L_c X L_q


class Pointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = Initialized_Conv1d(config.enc_filters * 2, 1)
        self.w2 = Initialized_Conv1d(config.enc_filters * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class Highway(nn.Module):
    def __init__(self, layer_num, size, config):
        super().__init__()
        self.n = layer_num
        self.config = config
        self.linear = nn.ModuleList(
            [
                Initialized_Conv1d(size, size, relu=False, bias=True)
                for _ in range(self.n)
            ]
        )
        self.gate = nn.ModuleList(
            [Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)]
        )

    def forward(self, x):
        # TODO: check why no non-linearity (see init of linear)
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(
                nonlinear, p=self.config.dropout, training=self.training
            )
            x = gate * nonlinear + (1 - gate) * x
            # x = F.relu(x)
        return x


class Embedding(nn.Module):
    """
    Computes word embedding 
    Steps:  
        - look up word
        - look up characters
        - if config.pre_trained is True:
            - pass chars through a 2D conv, kernel size is (1,5) so that we only convolve within word and not across word
        - do a max across the character dimension for character embedding
        - concatenate word and char embedding and pass through a highway unit
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.pretrained_char:
            self.conv2d = nn.Conv2d(
                config.char_emb_dim,
                config.char_emb_dim_projection,
                kernel_size=(1, 5),
                padding=0,
                bias=True,
            )
            nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity="relu")
            self.highway = Highway(
                2, config.word_emb_dim + config.char_emb_dim_projection, config
            )
            self.projection = Initialized_Conv1d(
                config.word_emb_dim + config.char_emb_dim_projection,
                config.enc_filters,
                relu=True,
                bias=True,
            )
            # # conv1d is more of a projection layer
            # # TODO: why no bias?
            # self.conv1d = Initialized_Conv1d(Dword + D, D, bias=False)
        else:
            self.highway = Highway(2, config.word_emb_dim + config.char_emb_dim, config)
            self.projection = Initialized_Conv1d(
                config.word_emb_dim + config.char_emb_dim,
                config.enc_filters,
                relu=True,
                bias=True,
            )

    def forward(self, char_emb, word_emb):

        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = F.dropout(
            char_emb, p=self.config.dropout_char, training=self.training
        )

        if self.config.pretrained_char:
            char_emb = F.relu(self.conv2d(char_emb))
        char_emb, _ = torch.max(char_emb, dim=3)

        word_emb = F.dropout(
            word_emb, p=self.config.dropout_word, training=self.training
        )
        word_emb = word_emb.transpose(1, 2)
        emb = torch.cat([char_emb, word_emb], dim=1)
        emb = self.highway(emb)
        emb = self.projection(emb)
        return emb


# TODO : As per paper we should retrain for OOV vectors. but we are not getting into that right now.
# Note about char and word embedding that we always load pre-trained word vectors but char may be trainable or not
class WordEmbedding(nn.Module):
    def __init__(self, wordmat, config):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.Tensor(wordmat), freeze=True)

    def forward(self, x):
        return self.emb(x)


class CharEmbedding(nn.Module):
    def __init__(self, charmat, config):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(
            torch.Tensor(charmat), freeze=config.pretrained_char
        )

    def forward(self, x):
        return self.emb(x)


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat, config):

        super().__init__()
        self.config = config

        self.word_emb = WordEmbedding(word_mat, config)
        self.char_emb = CharEmbedding(char_mat, config)

        # embedding layer
        self.emb = Embedding(config)

        # embedding encoder block
        self.emb_enc = EncoderBlock(
            conv_num=4, ch_num=config.enc_filters, k=7, config=config
        )

        self.cq_att = CQAttention(config)
        self.cq_resizer = Initialized_Conv1d(config.enc_filters * 4, config.enc_filters)
        self.model_enc_blks = nn.ModuleList(
            [
                EncoderBlock(conv_num=2, ch_num=config.enc_filters, k=7, config=config)
                for _ in range(7)
            ]
        )
        self.pointer_net = Pointer(config)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        result = self._forward(Cwid, Ccid, Qwid, Qcid)
        return result["start"], result["end"]

    def _forward(self, Cwid, Ccid, Qwid, Qcid):
        # compute masks first
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()

        # compute embeddings
        # word embeddings are 300 dim
        # char embeddings are 16 (chars) x200 dim
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)

        # embeddings encoder
        # output is same size as input
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)

        # Context query attention
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)

        # pass through encoder blocks
        M = F.dropout(M0, p=self.config.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M = blk(M, maskC, i * (2 + 2) + 1, 7)
        M1 = M

        M = F.dropout(M, p=self.config.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M = blk(M, maskC, i * (2 + 2) + 1, 7)
        M2 = M

        M = F.dropout(M, p=self.config.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M = blk(M, maskC, i * (2 + 2) + 1, 7)
        M3 = M

        p1, p2 = self.pointer_net(M1, M2, M3, maskC)
        return {
            "start": p1,
            "end": p2,
            "M0": M0,
            "M1": M1,
            "M2": M2,
            "M3": M3,
            "CQ_A": X,
            "emb_enc_C": Ce,
            "emb_enc_Q": Qe,
            "emb_C": C,
            "emb_Q": Q,
            "mask_C": maskC,
            "mask_Q": maskQ,
        }


class QANetV0(QANet):
    # random output for answerability
    def forward(self, Cwid, Ccid, Qwid, Qcid):
        p1, p2 = super().forward(Cwid, Ccid, Qwid, Qcid)
        batch_size = p1.size(0)
        fake_z = torch.rand(batch_size, device=p1.device)
        return p1, p2, fake_z


class QANetV1(QANet):
    def __init__(self, word_mat, char_mat, config):
        super().__init__(word_mat, char_mat, config)
        # See https://arxiv.org/pdf/1706.04115.pdf, page 5
        # bias for start and end
        bias = torch.empty(1)
        nn.init.uniform_(bias, -1, 1)
        self.start_bias = nn.Parameter(bias)
        bias = torch.empty(1)
        nn.init.uniform_(bias, -1, 1)
        self.end_bias = nn.Parameter(bias)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        p1, p2 = super().forward(Cwid, Ccid, Qwid, Qcid)
        p1 = torch.cat([p1, self.start_bias.expand([p1.shape[0], 1])], dim=1)
        p2 = torch.cat([p2, self.end_bias.expand([p2.shape[0], 1])], dim=1)

        # dummy return for z
        return p1, p2, torch.zeros((p1.size(0)), device=p1.device)


class QANetV2(QANet):
    def __init__(self, word_mat, char_mat, config):
        super().__init__(word_mat, char_mat, config)
        # start with all zero weights!
        # attention vector for z
        self.z_attn_w = Initialized_Conv1d(config.enc_filters * 3, 1)
        self.out_z = nn.Linear(config.enc_filters * 7, 1)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        result = self._forward(Cwid, Ccid, Qwid, Qcid)
        p1 = result["start"]
        p2 = result["end"]
        M1, M2, M3 = result["M1"], result["M2"], result["M3"]
        mask_C = result["mask_C"]

        # X1, X2 are used to calculate p1, p2
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        X3 = torch.cat([M1, M2, M3], dim=1)
        
        p3 = mask_logits(self.z_attn_w(X3).squeeze(), mask_C)

        p1_ = F.softmax(p1, dim=1)
        p2_ = F.softmax(p2, dim=1)
        p3_ = F.softmax(p3, dim=1)
        
        X1 = torch.sum(p1_.unsqueeze(1) * X1, dim=2)
        X2 = torch.sum(p2_.unsqueeze(1) * X2, dim=2)
        X3 = torch.sum(p3_.unsqueeze(1) * X3, dim=2)
        
        z = self.out_z(torch.cat([X1, X2, X3], dim=1))
        return p1, p2, z
