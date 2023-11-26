"""
1.74 GMac 787.15 k
"""
import torch
import torch.nn as nn


class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm((width, numUnits), eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm((width, numUnits), eps=1e-8)
    
    def forward(self,x):
        # x: (B, C, T, F)
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.channel) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.channel) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class DPCRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_conv1 = nn.Sequential(nn.ConstantPad2d([2,2,1,0], 0),
                                      nn.Conv2d(2, 32, (2,5), (1,2)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.en_conv2 = nn.Sequential(nn.ConstantPad2d([1,1,1,0], 0),
                                      nn.Conv2d(32, 32, (2,3), (1,2)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.en_conv3 = nn.Sequential(nn.ConstantPad2d([1,1,1,0], 0),
                                      nn.Conv2d(32, 32, (2,3), (1,2)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.en_conv4 = nn.Sequential(nn.ConstantPad2d([1,1,1,0], 0),
                                      nn.Conv2d(32, 64, (2,3), (1,1)),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU())
        self.en_conv5 = nn.Sequential(nn.ConstantPad2d([1,1,1,0], 0),
                                      nn.Conv2d(64, 128, (2,3), (1,1)),
                                      nn.BatchNorm2d(128),
                                      nn.PReLU())        
        self.dprnn1 = DPRNN(128, 33, 128)
        self.dprnn2 = DPRNN(128, 33, 128)

        self.de_conv5 = nn.Sequential(nn.ConvTranspose2d(256, 64, (2,3), (1,1)),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU())
        self.de_conv4 = nn.Sequential(nn.ConvTranspose2d(128, 32, (2,3), (1,1)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.de_conv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, (2,3), (1,2)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.de_conv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, (2,3), (1,2)),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU())
        self.de_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 2, (2,5), (1,2)),
                                      nn.BatchNorm2d(2))  

    def forward(self, x):
        """
        x: (B,F,T,2)
        """
        x_ref = x
        x = x.permute(0, 3, 2, 1)  # (B,C,T,F)
        en_x1 = self.en_conv1(x)      # ; print(en_x1.shape)
        en_x2 = self.en_conv2(en_x1)  # ; print(en_x2.shape)
        en_x3 = self.en_conv3(en_x2)  # ; print(en_x3.shape)
        en_x4 = self.en_conv4(en_x3)  # ; print(en_x4.shape)
        en_x5 = self.en_conv5(en_x4)  # ; print(en_x5.shape)
   
        en_xr = self.dprnn1(en_x5)    # ; print(en_xr.shape)
        en_xr = self.dprnn2(en_xr)    # ; print(en_xr.shape)
 
        de_x5 = self.de_conv5(torch.cat([en_x5, en_xr], dim=1))[...,:-1,:-2]  #; print(de_x5.shape)
        de_x4 = self.de_conv4(torch.cat([en_x4, de_x5], dim=1))[...,:-1,:-2]  #; print(de_x4.shape)
        de_x3 = self.de_conv3(torch.cat([en_x3, de_x4], dim=1))[...,:-1,:-2]  #; print(de_x3.shape)
        de_x2 = self.de_conv2(torch.cat([en_x2, de_x3], dim=1))[...,:-1,:-2]  #; print(de_x2.shape)
        de_x1 = self.de_conv1(torch.cat([en_x1, de_x2], dim=1))[...,:-1,:-4]  #; print(de_x1.shape)

        m = de_x1.permute(0,3,2,1)

        s_real = x_ref[...,0] * m[...,0] - x_ref[...,1] * m[...,1]
        s_imag = x_ref[...,1] * m[...,0] + x_ref[...,0] * m[...,1]
        s = torch.stack([s_real, s_imag], dim=-1)  # (B,F,T,2)
        
        return s


if __name__ == "__main__":
    model = DPCRN().cuda()

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print(flops, params)

    model = model.cpu().eval()
    x = torch.randn(1, 257, 63, 2)
    y = model(x)
    print(y.shape)