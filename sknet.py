import torch
from torch import nn

bn_momentum=0.1
class SKConv1 (nn.Module):
    def __init__(self, features,out_channel, M, G, r, stride=1, L=32):
        super (SKConv1, self).__init__ ()
        d = max (int (features / r), L)
        self.M = M
        self.features = features
        self.outchannel=out_channel

        self.conv1=nn.Sequential (
            nn.Conv2d (features, out_channel, kernel_size=3, stride=stride, padding=1,groups=G,bias=False),
            nn.BatchNorm2d (out_channel,momentum=bn_momentum),
            nn.ReLU (inplace=True)
        )
        self.conv2=nn.Sequential (
            nn.Conv2d (features, out_channel, kernel_size=3, stride=stride, padding=2,groups=G,bias=False,dilation=2),
            nn.BatchNorm2d (out_channel,momentum=bn_momentum),
            nn.ReLU (inplace=True)
        )
        # self.gap = nn.AvgPool2d (int(WH/stride))
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential (
                nn.Conv2d (out_channel, d, 1, padding=0,bias=False),
                # nn.BatchNorm2d (d),
                nn.ReLU (inplace=True)
            )
        self.fc2=nn.Sequential (
                nn.Conv2d (d, out_channel*2, 1, padding=0,bias=False),
                # nn.BatchNorm2d (256),
                nn.ReLU (inplace=True)
            )
        self.softmax = nn.Softmax (dim=1)

    def forward(self, x):
        fea1=self.conv1(x)
        fea2=self.conv2(x)
        fea_U = fea1+fea2
        fea_s = self.gap (fea_U)
        fea_z = self.fc1 (fea_s)
        fea_z=self.fc2(fea_z)
        fea_z=fea_z.view(fea_z.shape[0],2,-1,fea_z.shape[-1])


        attention_vectors = self.softmax (fea_z)
        attention_vectors1,attention_vectors2=torch.split(attention_vectors,1,dim=1)

        attention_vectors1=attention_vectors1.reshape(attention_vectors1.shape[0],self.outchannel,-1,attention_vectors1.shape[-1])
        attention_vectors2=attention_vectors2.reshape(attention_vectors2.shape[0],self.outchannel,-1,attention_vectors2.shape[-1])
        out1 = attention_vectors1*fea1
        out2 = attention_vectors2*fea2
        out=out1+out2
        return out

class SKConv2(nn.Module):
    def __init__(self, channel, reduction):
        super(SKConv2, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=2, dilation=2, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_ex = nn.Sequential(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        conv1 = self.conv1(x).unsqueeze(dim=1)
        conv2 = self.conv2(x).unsqueeze(dim=1)
        features = torch.cat([conv1, conv2], dim=1)
        U = torch.sum(features, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat([self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z).unsqueeze(dim=1)], dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (features * attention_vector).sum(dim=1)
        return V


class SKConv (nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super (SKConv, self).__init__ ()
        d = max (int (features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList ([])
        for i in range (M):
            self.convs.append (nn.Sequential (
                nn.Conv2d (features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d (features),
                nn.ReLU (inplace=False)
            ))
        self.gap = nn.AvgPool2d (int (WH / stride))
        self.fc=nn.Conv2d (features, d, 1, padding=0, bias=False),
        self.relu=nn.ReLU (inplace=True)
        # self.fc = nn.Linear (features, d)
        self.fc=nn.Sequential(
            nn.Conv2d (features, d, 1, padding=0, bias=False),
            nn.BatchNorm2d (d),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList ([])
        for i in range (M):
            self.fcs.append (
                nn.Sequential(
                # nn.Linear (d, features)
                nn.Conv2d (d, features, 1, padding=0, bias=False),
                nn.BatchNorm2d (features),
                nn.ReLU(inplace=True))
            )
        self.softmax = nn.Softmax (dim=1)

    def forward(self, x):
        for i, conv in enumerate (self.convs):
            fea = conv (x).unsqueeze_ (dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat ([feas, fea], dim=1)
        fea_U = torch.sum (feas, dim=1)
        fea_s = self.gap (fea_U).squeeze_ ()
        print(fea_s.shape)
        fea_s_in=fea_s.view(fea_s.shape[0],fea_s.shape[1],1,1)
        fea_z = self.fc (fea_s_in)
        # fea_z = self.relu(self.fc (fea_s))
        for i, fc in enumerate (self.fcs):
            vector = fc (fea_z).unsqueeze_ (dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat ([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax (attention_vectors)
        attention_vectors = attention_vectors#.unsqueeze (-1)#.unsqueeze (-1)
        print(attention_vectors.shape)
        fea_v = (feas * attention_vectors).sum (dim=1)
        return fea_v


class SKUnit (nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super (SKUnit, self).__init__ ()
        if mid_features is None:
            mid_features = int (out_features / 2)
        self.feas = nn.Sequential (
            nn.Conv2d (in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d (mid_features),
            SKConv (mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d (mid_features),
            nn.Conv2d (mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d (out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential ()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential (
                nn.Conv2d (in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d (out_features)
            )

    def forward(self, x):
        fea = self.feas (x)
        return fea + self.shortcut (x)





if __name__ == '__main__':
    dump_input = torch.rand ((64,
                              128,
                              64,
                              64))

    sk=SKConv(128,128,2,8,2)
    # sk=SKAttention(64,2)
    res=sk(dump_input)
    print(res.shape)