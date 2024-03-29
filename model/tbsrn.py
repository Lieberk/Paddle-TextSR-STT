import copy
import math
import warnings

import paddle
import paddle.nn.functional as F
from paddle import nn
from utils.util import masked_fill
warnings.filterwarnings("ignore")
from model.tps_spatial_transformer import TPSSpatialTransformer
from model.stn_head import STNHead


def clones(module, N):
    "Produce N identical layers."
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Layer):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = paddle.create_parameter(shape=[features, ], dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=1.))
        self.b_2 = paddle.create_parameter(shape=[features, ], dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=0.))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = paddle.zeros([d_model, height, width])
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = paddle.exp(paddle.arange(0., d_model, 2) *
                          -(math.log(10000.0) / d_model))
    pos_w = paddle.arange(0., width, dtype='float32').unsqueeze(1)
    pos_h = paddle.arange(0., height, dtype='float32').unsqueeze(1)

    pe[0:d_model:2, :, :] = paddle.sin(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    pe[1:d_model:2, :, :] = paddle.cos(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    pe[d_model::2, :, :] = paddle.sin(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])
    pe[d_model + 1::2, :, :] = paddle.cos(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])

    return pe


class FeatureEnhancer(nn.Layer):

    def __init__(self):
        super(FeatureEnhancer, self).__init__()

        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128, 64)

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch = conv_feature.shape[0]
        position2d = positionalencoding2d(64, 16, 64).cast('float32').unsqueeze(0).reshape([1, 64, 1024])
        position2d = position2d.tile([batch, 1, 1])
        conv_feature = paddle.concat([conv_feature, position2d], 1)  # batch, 128(64+64), 32, 128
        result = conv_feature.transpose([0, 2, 1])
        origin_result = result
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)
        return result.transpose([0, 2, 1])


class MultiHeadedAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).reshape([nbatches, -1, self.h, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose([0, 2, 1, 3]).reshape([nbatches, -1, self.h * self.d_k])

        return self.linears[-1](x), attention_map


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.shape[-1]
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = masked_fill(scores, mask == 0, float('-inf'))
    else:
        pass
    p_attn = F.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return paddle.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Layer):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TBSRN(nn.Layer):
    def __init__(self, scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=False, hidden_units=32, input_channel=3):
        super(TBSRN, self).__init__()

        self.conv = nn.Conv2D(input_channel, 3, 3, 1, 1)
        self.bn = nn.BatchNorm2D(3)
        self.relu = nn.ReLU()

        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2D(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2D(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2D(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2D(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = paddle.tanh(block[str(self.srb_nums + 3)])
        return output


class RecurrentResidualBlock(nn.Layer):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(channels)
        self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancer()

        for p in self.parameters():
            if p.dim() > 1:
                paddle.nn.initializer.XavierUniform(p)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        residual = residual.reshape([size[0], size[1], -1])
        residual = self.feature_enhancer(residual)
        residual = residual.reshape([size[0], size[1], size[2], size[3]])
        return x + residual


class UpsampleBLock(nn.Layer):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2D(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Layer):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (paddle.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, direction='bidirectional')

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # b, w, h, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
