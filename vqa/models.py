"""
VQA models.

BaselineNet featurizes the image and the question into 1-d vectors.
Then concatenates the two representations and feeds to a linear layer.
The backbones are frozen.

TransformerNet featurizes image-language into sequences,
then applies iterative self-/cross-attention on them.
Backbones are again frozen.
"""

import math

import torch
from torch import nn
from torchvision.models.resnet import resnet18
from transformers import RobertaModel, RobertaTokenizerFast


class BaselineNet(nn.Module):
    """Simple baseline for VQA."""

    def __init__(self, n_answers=5217):
        """Initialize layers given the number of answers."""
        super().__init__()
        # Text encoder
        t_type = "roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Visual encoder
        r18 = resnet18(pretrained=True)
        self.vis_encoder = nn.Sequential(*(list(r18.children())[:-2]))
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        # Classifier
            # torch.Size([1, 768]) torch.Size([1, 512]) 
        n_hidden = 1024
        self.classifier = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + 512, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_answers)
        )

    def forward(self, image, question):
        """Forward pass, image (B, 3, 224, 224), qs list of str."""
        text_feat = self.compute_text_feats(question)
        im_feat = self.compute_vis_feats(image)
        # torch.Size([1, 768]) torch.Size([1, 512]) >  [1, 1280]
        return self.classifier(torch.cat((text_feat, im_feat), axis=1))

    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        return encoded_text.pooler_output

    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        image = self.vis_encoder(image)  
        # feed to vis_encoder and then mean pool on spatial dims
        return torch.nn.AvgPool2d(image.size()[-1])(image).squeeze(-1).squeeze(-1)

    def train(self, mode=True):
        """Override train to set backbones in eval mode."""
        nn.Module.train(self, mode=mode)
        self.vis_encoder.eval()
        self.text_encoder.eval()

class LSTMNet(BaselineNet):
    def __init__(self, n_answers=5217, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        #embedding = nn.Embedding(voc_size, emb_dim)
        self.im_feat_dim = 512
        self.text_feat_dim = 768
        self.conv1d = nn.Conv1d(self.im_feat_dim, self.text_feat_dim,
                                kernel_size=1, stride=1, padding=0)
        self.text_rnn = nn.LSTM(input_size=self.text_feat_dim, hidden_size=self.text_feat_dim,
                                bidirectional=True, num_layers=n_layers, batch_first=True)
        self.classifier = nn.Linear(2 * n_layers * self.text_feat_dim, n_answers)
    
    def forward(self, image, question):
        # get visual features
        text_feat = self.compute_text_feats(question)
        im_feat = self.compute_vis_feats(image)
        _, L, H = text_feat.size()
        #
        im_feat = self.conv1d(im_feat.T).T
        im_feat = torch.stack([im_feat]*2*self.n_layers, dim=0)
        # Im feat:torch.Size([4, *, 768])
        out, (h_out, _) =  self.text_rnn(text_feat, (im_feat, im_feat))
        h_out = h_out.permute(1, 0, 2).flatten(1)
        # torch.Size([*, L=12, 768*2=1536]) torch.Size([1, 3072])
        return self.classifier(h_out)
    
    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        image = self.vis_encoder(image)  
        # feed to vis_encoder and then mean pool on spatial dims
        ks = image.size()[-1]
        return torch.nn.AvgPool2d(ks)(image).squeeze(-1).squeeze(-1)
    
    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        return encoded_text.last_hidden_state
    
class LSTMTransfromer(BaselineNet):
    def __init__(self, n_answers=5217, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        #embedding = nn.Embedding(voc_size, emb_dim)
        self.im_feat_dim = 512
        self.text_feat_dim = 768
        self.conv1d = nn.Conv1d(self.im_feat_dim, self.text_feat_dim,
                                kernel_size=1, stride=1, padding=0)
        self.text_rnn = nn.LSTM(input_size=self.text_feat_dim, hidden_size=self.text_feat_dim,
                                bidirectional=True, num_layers=n_layers, batch_first=True)
        
        # Project to 256
        self.text_proj = nn.Linear(self.text_feat_dim, 256)  # 768 -> 256
        self.vis_proj = nn.Linear(512, 256)

        # Cross-attention
        self.ca = nn.ModuleList([CrossAttentionLayer(256) for _ in range(3)])
        
    
    def forward(self, image, question):
        # get visual features
        text_feat = self.compute_text_feats(question)
        im_feat = self.compute_vis_feats(image)
        _, L, H = text_feat.size()
        #
        im_feat = self.conv1d(im_feat.T).T
        im_feat = torch.stack([im_feat]*2*self.n_layers, dim=0)
        # Im feat:torch.Size([4, *, 768])
        out, (lstm_out, _) =  self.text_rnn(text_feat, (im_feat, im_feat))
        lstm_out = lstm_out.permute(1, 0, 2).flatten(1)
        # torch.Size([*, L=12, 768*2=1536]) torch.Size([1, 3072])
        
        for layer in self.ca:
            vis_feats, text_feats = layer(
                vis_feats, None,
                text_feats, text_mask,
                seq1_pos=vis_pos
            )
        # (*, S2, F) > # (*, F)
        text_feats = (text_feats * (~text_mask)[..., None].float()).sum(1) / \
            (~text_mask)[..., None].float().sum(1)
        # (*, S2, F) > # (*, F)
        vis_feats = vis_feats.mean(1)
        # Classifier
        # 512 > n_answers
        return self.classifier(torch.cat((text_feats, vis_feats), 1))
    
    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        image = self.vis_encoder(image)  
        # feed to vis_encoder and then mean pool on spatial dims
        ks = image.size()[-1]
        return torch.nn.AvgPool2d(ks)(image).squeeze(-1).squeeze(-1)
    
    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        return encoded_text.last_hidden_state

class TransformerNet(BaselineNet):
    """Simple transformer-based model for VQA."""

    def __init__(self, n_answers=5217):
        """Initialize layers given the number of answers."""
        super().__init__()
        # Text/visual encoders are inhereted from BaselineNet
        # Positional embeddings
        self.pos_embed = PositionEmbeddingSine(128, normalize=True)

        # Project to 256
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 256)  # 768 -> 256
        self.vis_proj = nn.Linear(512, 256)

        # Cross-attention
        self.ca = nn.ModuleList([CrossAttentionLayer(256) for _ in range(3)])

        # Classifier
        self.classifier = nn.Linear(256 + 256, n_answers)

    def forward(self, image, question):
        """Forward pass, image (B, 3, 224, 224), qs list of str."""
        # Per-modality encoders
        text_feats, text_mask = self.compute_text_feats(question)
        text_feats = self.text_proj(text_feats)
        # (*, L, 256)
        vis_feats, vis_pos = self.compute_vis_feats(image)
        vis_feats = self.vis_proj(vis_feats)
        # (*, F=7x7=49, 256)
        vis_pos = vis_pos.to(vis_feats.device)
        
        #print(text_feats.size(), text_mask.size(), vis_feats.size(), vis_pos.size())
        # text: torch.Size([*, 18, 256]) torch.Size([*, 18]) 
        # vis: torch.Size([*, 49, 256]) torch.Size([1, 49, 256])
        # Cross-encoder
        for layer in self.ca:
            # seq1, seq1_key_padding_mask, seq2, seq2_key_padding_mask, seq1_pos=None, seq2_pos=None
            vis_feats, text_feats = layer(
                vis_feats, None,
                text_feats, text_mask,
                seq1_pos=vis_pos
            )
        
        # (*, S2, F) > # (*, F)
        text_feats = (text_feats * (~text_mask)[..., None].float()).sum(1) / (~text_mask)[..., None].float().sum(1)
        # (*, S2, F) > # (*, F)
        vis_feats = vis_feats.mean(1)
        # Classifier
        # 512 > n_answers
        return self.classifier(torch.cat((text_feats, vis_feats), 1))

    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        return encoded_text.last_hidden_state, text_attention_mask

    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        encoded_img = self.vis_encoder(image)
        # (*, 512, 7, 7) -> (*, 512, 49)
        B, F, D, _ = encoded_img.shape
        # (*, 49, 512), (1, 49, 128*2)
        return (
            encoded_img.reshape(B, F, D * D).transpose(1, 2),
            self.pos_embed(D).reshape(1, -1, D * D).transpose(1, 2)
        )

class CrossAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # Self-attention for seq1
        self.sa1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )  # use batch_first=True everywhere!
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        self.sa2 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.cross_12 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_12 = nn.Dropout(dropout)
        self.norm_12 = nn.LayerNorm(d_model)

        # FFN for seq1
        n_hidden = 1024    # hidden dim is 1024
        self.ffn_12 = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm_122 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.cross_21 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_21 = nn.Dropout(dropout)
        self.norm_21 = nn.LayerNorm(d_model)

        # FFN for seq2
        self.ffn_21 = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm_212 = nn.LayerNorm(d_model)

    def forward(self, seq1, seq1_key_padding_mask,
                seq2, seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        # torch.Size([*, 18, 256]) torch.Size([*, 18]) 
        # # vis: torch.Size([*, 49, 256]) torch.Size([1, 49, 256])
        # Self-attention for seq1
        q1 = k1 = v1 = seq1
        if seq1_pos is not None:
            q1 = q1 + seq1_pos
            k1 = k1 + seq1_pos
        seq1b = self.sa1(
            query=q1,
            key=k1,
            value=v1,
            attn_mask=None,
            key_padding_mask=seq1_key_padding_mask  # (B, S1)
        )[0]
        seq1 = self.norm_1(seq1 + self.dropout_1(seq1b))
        # (* , S1, S1) - att > (*, S1, F) 
        # Self-attention for seq2
        q2 = k2 = v2 = seq2
        if seq2_pos is not None:
            q2 = q2 + seq2_pos
            k2 = k2 + seq2_pos
        seq2b = self.sa2(
            query=q2,
            key=k2,
            value=v2,
            attn_mask=None,
            key_padding_mask=seq2_key_padding_mask  # (B, S1)
        )[0]
        seq2 = self.norm_1(seq2 + self.dropout_1(seq2b))
        # (*, S2, F)
        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = seq1
        q2 = k2 = v2 = seq2
        if seq1_pos is not None:
            q1 = q1 + seq1_pos
            k1 = k1 + seq1_pos
        if seq2_pos is not None:
            q2 = q2 + seq2_pos
            k2 = k2 + seq2_pos
        
        # Cross-attention from seq1 to seq2 and FFN
        cross12_attn, _ = self.cross_12(q1, k2, v2,
                                    key_padding_mask=seq2_key_padding_mask)
        cross12_attn_out = self.norm_12(self.dropout_12(cross12_attn)+seq1)
        # (*, S1, F)
        # Cross-attention from seq2 to seq1 and FFN
        cross21_attn, _ = self.cross_21(q2, k1, v1,
                                    key_padding_mask=seq1_key_padding_mask)
        cross21_attn_out = self.norm_21(self.dropout_21(cross21_attn)+seq2)
        # (*, S2, F)
            # ffn
        seq1_attn_out = self.norm_122(self.ffn_12(cross12_attn_out)+cross12_attn_out)
        seq2_attn_out = self.norm_212(self.ffn_21(cross21_attn_out)+cross21_attn_out)
        return seq1_attn_out, seq2_attn_out


class PositionEmbeddingSine(nn.Module):
    """
    2-D version of position embeddings,
    similar to the "Attention is all you need" paper.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, spat_dim):
        mask = torch.zeros(1, spat_dim, spat_dim).bool()
        # (1, 49, 49) of True
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # (self.num_pos_feats = 128)
        # torch.Size([1, 49, 49, 1]) /  [1, 128] > (Broadcasted) > [1,49,49,128]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # torch.Size([1, 49, 49, 128]) 
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # torch.Size([1, 256, 49, 49])
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
