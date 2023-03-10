from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 8, dropout = 0.):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x, mask = None) :
        # split keys, queries and values in num_heads
        #print("1qkv's shape: ", self.qkv(x).shape)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        #print("2qkv's shape: ", qkv.shape)
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        #print("queries's shape: ", queries.shape)
        #print("keys's shape: ", keys.shape)
        #print("values's shape: ", values.shape)
        
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        #print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        #print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        #print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        #print("att2' shape: ", att.shape)
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        #print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        #print("out2's shape: ", out.shape)
        out = self.projection(out)
        #print("out3's shape: ", out.shape)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self,emb_size, expansion = 4, drop_p = 0.):
        super(FeedForwardBlock,self).__init__()
        self.FC1 = nn.Linear(emb_size, expansion * emb_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.FC2 = nn.Linear(expansion * emb_size, emb_size)

    def forward(self,x):
        x = self.FC1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.FC2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size = 768,drop_p = 0.,forward_expansion = 4,forward_drop_p = 0.):
        super(TransformerEncoderBlock,self).__init__()
        self.LN1 = nn.LayerNorm(emb_size)
        self.multiheadattention = MultiHeadAttention(emb_size = 768, num_heads = 8, dropout = 0.)
        self.drop_out1 = nn.Dropout(drop_p)

        self.LN2 = nn.LayerNorm(emb_size)
        self.FF = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.drop_out2 = nn.Dropout(drop_p)

    def forward(self,x):
        res1 = x
        x = self.LN1(x)
        x = self.multiheadattention(x)
        x = self.drop_out1(x)
        x += res1

        res2 = x
        x = self.LN2(x)
        x = self.FF(x)
        x = self.drop_out2(x)
        x += res2

        return x



if __name__ == '__main__':
    from PIL import Image
    img = Image.open('./man.jpg')
    transform = Compose([Resize((224, 224)), ToTensor()])
    x = transform(img)
    x = x.unsqueeze(0)

    patch_size = 14 # 16 pixels
    pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)#长和宽都分了16份，每一份的面积为14*14=196
    print(pathes.shape)

    class PatchEmbedding4(nn.Module):
        def __init__(self, in_channels: int = 3, patch_size: int = 14, emb_size: int = 768, img_size: int = 224):
            self.patch_size = patch_size
            super().__init__()
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
            # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
            self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))
        
        def forward(self, x: Tensor) -> Tensor:
            b, _, _, _ = x.shape
            x = self.projection(x)
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
            # prepend the cls token to the input
            x = torch.cat([cls_tokens, x], dim=1)
            # add position embedding
            #print(x.shape, self.positions.shape)
            x += self.positions#位置编码上加到每张图片上
            return x

    patches_embedded = PatchEmbedding4()(x)
    print(patches_embedded.shape)

    layer = TransformerEncoderBlock(emb_size = 768,drop_p = 0.,forward_expansion = 4,forward_drop_p = 0.)
    patches_embedded = layer(patches_embedded)
    print(patches_embedded.shape)