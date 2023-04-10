import torch
import torch.nn as nn
import timm
from models.pet_modules import VanillaEncoder, AdaptFormer

class AVmodel(nn.Module):
    def __init__(self, num_classes, num_latents, dim):
        super(AVmodel, self).__init__()

        self.v1 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True) # for spectrogram (to be updated with AST weights)
        self.v2 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True) # for RGB images

        """
        discard unnecessary layers and save parameters
        """
        self.v1.pre_logits = nn.Identity()
        self.v2.pre_logits = nn.Identity()
        self.v1.head = nn.Identity()
        self.v2.head = nn.Identity()
        
        """
        update v1 weights with that of AST (Audio Spectrogram Transformer)
        https://github.com/YuanGongND/ast 
        """
        # use weight "Full AudioSet, 16 tstride, 16 fstride, without Weight Averaging, Model (0.442 mAP)"
        ast_pretrained_weight = torch.load("audioset_16_16_0.4422.pth?dl=1") 

        # replace CNN for spec with 1d input channel
        self.v1.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.v1.pos_embed = nn.Parameter(torch.randn(1, 514, 768) * .02) # 514 pos embed tokens in AST

        # initial weights
        v = self.v1.state_dict()

        # update weights
        v['cls_token'] = ast_pretrained_weight['module.v.cls_token']
        v['pos_embed'] = ast_pretrained_weight['module.v.pos_embed']
        v['patch_embed.proj.weight'] = ast_pretrained_weight['module.v.patch_embed.proj.weight']
        v['patch_embed.proj.bias'] = ast_pretrained_weight['module.v.patch_embed.proj.bias']
        for i in range(12):
            v['blocks.'+str(i)+'.norm1.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.norm1.weight']
            v['blocks.'+str(i)+'.norm1.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.norm1.bias']
            v['blocks.'+str(i)+'.attn.qkv.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight']
            v['blocks.'+str(i)+'.attn.qkv.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias']
            v['blocks.'+str(i)+'.attn.proj.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.attn.proj.weight']
            v['blocks.'+str(i)+'.attn.proj.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.attn.proj.bias']
            v['blocks.'+str(i)+'.norm2.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.norm2.weight']
            v['blocks.'+str(i)+'.norm2.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.norm2.bias']
            v['blocks.'+str(i)+'.mlp.fc1.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.mlp.fc1.weight']
            v['blocks.'+str(i)+'.mlp.fc1.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.mlp.fc1.bias']
            v['blocks.'+str(i)+'.mlp.fc2.weight'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.mlp.fc2.weight']
            v['blocks.'+str(i)+'.mlp.fc2.bias'] = ast_pretrained_weight['module.v.blocks.'+str(i)+'.mlp.fc2.bias']
        v['norm.weight'] = ast_pretrained_weight['module.v.norm.weight']
        v['norm.bias'] = ast_pretrained_weight['module.v.norm.bias']

        # load weights
        self.v1.load_state_dict(v)

        """
        Freeze parameters
        """
        # Trainable by default = Spec+RGB pos embed and cls token, linear classifier
        
        # spec
        self.v1.pos_embed.requires_grad=False
        for p in self.v1.patch_embed.proj.parameters():p.requires_grad=False
        for p in self.v1.blocks.parameters():p.requires_grad=False

        # RGB
        self.v2.pos_embed.requires_grad=False
        for p in self.v2.patch_embed.proj.parameters():p.requires_grad=False
        for p in self.v2.blocks.parameters():p.requires_grad=False                   

        """
        Initialize conv projection, cls token, pos embed and encoders for audio and visual modality
        """
        # conv projection
        self.spec_conv = self.v1.patch_embed.proj
        self.rgb_conv = self.v2.patch_embed.proj

        # cls token and pos embedding
        self.spec_pos_embed = self.v1.pos_embed
        self.rgb_pos_embed = self.v2.pos_embed

        self.spec_cls_token = self.v1.cls_token
        self.rgb_cls_token = self.v2.cls_token

        """
        Initialize Encoder, Final Norm and Classifier
        """
        encoder_layers = []
        for i in range(12):

            # Vanilla Transformer Encoder (use for full fine tuning)
            # encoder_layers.append(VanillaEncoder(num_latents=num_latents, spec_enc=self.v1.blocks[i], rgb_enc=self.v2.blocks[i]))

            # Frozen Transformer Encoder with AdaptFormer 
            encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim, spec_enc=self.v1.blocks[i], rgb_enc=self.v2.blocks[i]))
             
        self.audio_visual_blocks = nn.Sequential(*encoder_layers)

        # final norm
        self.spec_post_norm = self.v1.norm
        self.rgb_post_norm = self.v2.norm

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(768,num_classes)
        )

        """
        Forward pass for Spectrogram and RGB Images
        """
    def forward_spec_features(self,x): # shape = (bs, 128, 1006)

        x = x.unsqueeze(1) # shape = (bs, 1, 128, 1006)
        x = self.spec_conv(x) # shape = (bs, 1, 128, 1006)
        B, dim, f_dim, t_dim = x.shape # shape = (bs, 768, 8, 62)
        x = torch.reshape(x,(B, dim, f_dim*t_dim)) # shape = (bs, 768, 496)
        x = x.permute(0,2,1) # shape = (bs, 496, 768)

        x = torch.cat((self.spec_cls_token.expand(B, -1, -1),x), dim=1) # shape = (bs, 1+496, 768)
        # interpolate pos embedding and add
        x = x + nn.functional.interpolate(self.spec_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1)
        return x
        
    def forward_rgb_features(self,x):
        B, no_of_frames, C, H, W = x.shape # shape = (bs, no_of_frames, 3, 224, 224)
        x = torch.reshape(x, (B * no_of_frames, C, H, W)) # shape = (bs*no_of_frames, 3, 224, 224)
        x = self.rgb_conv(x) # shape = (bs*no_of_frames, 768, 14, 14)

        _, dim, h, w = x.shape
        x = torch.reshape(x, (B, no_of_frames, dim, h, w)) # shape = (bs, no_of_frames, 768, 14, 14)
        x = x.permute(0,2,1,3,4) # shape = (bs, 768, no_of_frames, 14, 14)
        x = torch.reshape(x, (B, dim, no_of_frames*h*w)) # shape = (bs, 768, no_of_frames*14*14) = (bs, 768, 1568)
        x = x.permute(0,2,1) # shape = (bs, 1568, 768); 1568 = spatio-temporal tokens for 8 RGB images

        x = torch.cat((self.rgb_cls_token.expand(B, -1, -1),x), dim=1) # shape = (bs, 1+1568, 768)
        # interplate pos embedding and add
        x = x + nn.functional.interpolate(self.rgb_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1)
        return x

    def forward_encoder(self,x,y):     
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            x,y = blk(x,y)

        x = self.spec_post_norm(x)
        y = self.rgb_post_norm(y)

        # return class token alone
        x = x[:, 0]
        y = y[:, 0]
        return x,y
        
    def forward(self, x, y):

        x = self.forward_spec_features(x)
        y = self.forward_rgb_features(y)
        x,y = self.forward_encoder(x,y)

        logits = (x+y)*0.5
        logits = self.classifier(logits)
        return logits