import torch
from torch import nn
from torch.nn import functional as F
from hsi_shadow.models.ssrn_official import RawQuotientDualStreamSSRN

class LocalSpectral(nn.Module):
    """Full physical band axis independently at each patch pixel."""
    def __init__(self):
        super().__init__()
        self.c3=nn.Conv1d(1,4,3,padding=1,padding_mode='replicate')
        self.c7=nn.Conv1d(1,4,7,padding=3,padding_mode='replicate')
        self.out=nn.Conv1d(8,1,1,bias=False)
        self.scale=nn.Parameter(torch.tensor(0.1))
    def forward(self,x):
        n,b,h,w=x.shape
        z=x.permute(0,2,3,1).reshape(-1,1,b)
        y=self.out(F.gelu(torch.cat([self.c3(z),self.c7(z)],dim=1)))
        return x+self.scale*y.reshape(n,h,w,b).permute(0,3,1,2)

class ChannelRelations(nn.Module):
    """Independent simplified channel attention; CIDNet-inspired."""
    def __init__(self,sparse=False,channels=128,heads=4):
        super().__init__()
        self.sparse=sparse;self.heads=heads
        self.norm=nn.GroupNorm(1,channels)
        self.qkv=nn.Conv2d(channels,channels*3,1,bias=False)
        self.out=nn.Conv2d(channels,channels,1,bias=False)
        self.temperature=nn.Parameter(torch.ones(heads,1,1))
        self.scale=nn.Parameter(torch.tensor(0.1))
    def weights(self,q,k):
        a=(F.normalize(q,dim=-1)@F.normalize(k,dim=-1).transpose(-1,-2))*self.temperature
        if self.sparse:
            idx=a.topk(a.shape[-1]//2,dim=-1).indices
            mask=torch.zeros_like(a,dtype=torch.bool).scatter_(-1,idx,True)
            a=a.masked_fill(~mask,float('-inf'))
        return a.softmax(-1)
    def forward(self,x):
        n,c,h,w=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(n,3,self.heads,c//self.heads,h*w).unbind(1)
        z=(self.weights(q,k)@v).reshape(n,c,h,w)
        return x+self.scale*self.out(z)

class Wrapper(nn.Module):
    def __init__(self,cfg,classes,bands,patch):
        super().__init__()
        self.core=RawQuotientDualStreamSSRN(bands,classes,patch_size=patch)
        self.kind=cfg['encoder']
        self.extra=(LocalSpectral() if self.kind=='local' else ChannelRelations(self.kind=='sparse') if self.kind in ('dense','sparse') else nn.Identity())
    def forward(self,x,illum=None,quotient=None):
        if quotient is None:raise ValueError('Auxiliary view required')
        q=self.extra(quotient) if self.kind=='local' else quotient
        raw=self.core._spectral_features(x);aux=self.core._quotient_features(q)
        if self.kind in ('dense','sparse'):aux=self.extra(aux)
        return self.core._classify_compressed(raw+self.core.quotient_projection(aux))

def build_model(cfg,num_classes,input_bands,patch_size):
    return Wrapper(cfg,num_classes,input_bands,patch_size)
def fit_scale(cfg,ds):
    # Dataset owns frozen-training-support statistics. No extra scale fitting.
    return None

