import paddle.nn as nn

cnt = 0


class BidirectionalLSTM(nn.Layer):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, direction='bidirectional', time_major=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.shape
        t_rec = recurrent.reshape([T * b, h])

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.reshape([T, b, -1])

        return output


class CRNN(nn.Layer):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_sublayer('conv{0}'.format(i),
                             nn.Conv2D(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_sublayer('batchnorm{0}'.format(i), nn.BatchNorm2D(nOut))
            if leakyRelu:
                cnn.add_sublayer('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            else:
                cnn.add_sublayer('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_sublayer('pooling{0}'.format(0), nn.MaxPool2D(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_sublayer('pooling{0}'.format(1), nn.MaxPool2D(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_sublayer('pooling{0}'.format(2),
                       nn.MaxPool2D((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_sublayer('pooling{0}'.format(3),
                       nn.MaxPool2D((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.shape
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.transpose([2, 0, 1])  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


if __name__ == '__main__':
    import paddle

    crnn = CRNN(32, 3, 37, 256)
    input = paddle.to_tensor([32, 3, 16, 64])
    output = crnn(input)
    print(output.shape)
