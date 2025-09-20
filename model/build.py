from torch.nn import TransformerEncoderLayer, TransformerEncoder
from utils.simple_tokenizer import SimpleTokenizer
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import random

#
class PredictionTextModel(nn.Module):
    def __init__(self, args, embed_dim):
        self.embed_dim = embed_dim
        super(PredictionTextModel, self).__init__()

        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)



        scale = self.embed_dim ** -0.5  # 使用 embed_dim 直接计算 scale

        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        # 删除了与 cross_modal_transformer 相关的初始化参数计算
        proj_std = scale * ((2 * 1) ** -0.5)  # 修改为固定值 1
        attn_std = scale
        # fc_std = (2 * self.embed_dim) ** -0.5  # 使用 embed_dim 直接计算 fc_std



        # 初始化 cross_attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def forward(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

class PredictionImgModel(nn.Module):
    def __init__(self, args, embed_dim):
        self.embed_dim = embed_dim
        super(PredictionImgModel, self).__init__()

        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)

        # 删除了 self.cross_modal_transformer 相关的代码
        # self.cross_modal_transformer = Transformer(width=self.embed_dim, layers=1, heads=self.embed_dim // 64)
        # scale = self.cross_modal_transformer.width ** -0.5

        scale = self.embed_dim ** -0.5  # 使用 embed_dim 直接计算 scale

        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        # 删除了与 cross_modal_transformer 相关的初始化参数计算
        proj_std = scale * ((2 * 1) ** -0.5)  # 修改为固定值 1
        attn_std = scale
        # fc_std = (2 * self.embed_dim) ** -0.5  # 使用 embed_dim 直接计算 fc_std

        # 删除了对 cross_modal_transformer.resblocks 的权重初始化
        # for block in self.cross_modal_transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化 cross_attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def forward(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x


class FCSA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.tokenizer = SimpleTokenizer()

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'stl' in args.loss_names :

            # self.textPre_model = PredictionTextModel(args, self.embed_dim)
            self.imagePre_model = PredictionImgModel(args, self.embed_dim)

            self.fc = nn.Linear(self.embed_dim, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            #
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=2,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)


    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_ = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x


    def mask_random_patches_with_noise(self, img, mask_indices, patch_size=16, stride_size=16, mask_fraction=0.3):
        _, c, h, w = img.unsqueeze(0).shape  # 获取图像的批量大小、通道数、高度和宽度
        num_patches_h = (h - patch_size) // stride_size + 1  # 图像在高度方向上被分成的块数
        num_patches_w = (w - patch_size) // stride_size + 1  # 图像在宽度方向上被分成的块数
        total_patches = num_patches_h * num_patches_w  # 总块数

        patches = F.unfold(img.unsqueeze(0), kernel_size=patch_size,
                           stride=stride_size)  # [1, C*patch_size*patch_size, L]
        patches = patches.squeeze(0).permute(1, 0).contiguous().view(-1, c, patch_size,
                                                                     patch_size)  # [L, C, patch_size, patch_size]

        # print(f'Total patches: {total_patches}, Patch size: {patch_size}x{patch_size}')

        for idx in mask_indices:
            patches[idx] = torch.randn_like(patches[idx])

        patches = patches.view(num_patches_h, num_patches_w, c, patch_size, patch_size).permute(2, 0, 3, 1,
                                                                                                4).contiguous().view(c,
                                                                                                                     h,
                                                                                                                     w)
        return patches


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x,_ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def _build_random_masked_tokens_and_labels(self, captions, sorted_t_indices):

        mask = self.tokenizer.encoder["<|mask|>"]

        labels = torch.zeros(captions.size(0), dtype=torch.int64).to('cuda')

        x = captions.argmax(dim=-1)

        # 计算 t 的值
        t = int((x + 1) * self.args.t_mask)

        filtered_indices = [idx.item() for idx in sorted_t_indices if 0 < idx.item() < x]
        # 获取 sorted_indices 的前 t 个元素
        selected_indices = filtered_indices[:t]

        labels[selected_indices] = captions[selected_indices]

        captions[selected_indices] = mask

        return captions, labels

    def forward(self, batch, epoch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']
        ps_images = batch['ps_images']
        ps_caption_ids = batch['ps_caption_ids']
        mask_tokens = batch['mlm_ids']
        ps_mask_tokens = batch['ps_mlm_ids']

        with torch.no_grad():

            img_feats = self.encode_image(images)
            txt_feats = self.encode_text(caption_ids)
            ps_imgs = ps_images.view(-1, 3, 384, 128)
            ps_caption = ps_caption_ids.view(-1, 77)
            ps_images_feats = self.encode_image(ps_imgs)
            ps_text_feats = self.encode_text(ps_caption)

            ps_images_feats = ps_images_feats.view(images.size(0), 4, -1)
            ps_text_feats = ps_text_feats.view(images.size(0), 4, -1)

            batch_indices = torch.arange(ps_images.shape[0])

            image_feats_unsqueezed = img_feats.unsqueeze(1)
            text_feats_unsqueezed = txt_feats.unsqueeze(1)

            cos_i2t_sim = F.cosine_similarity(image_feats_unsqueezed, ps_text_feats, dim=-1)
            cos_t2i_sim = F.cosine_similarity(text_feats_unsqueezed, ps_images_feats, dim=-1)

            _, min_i2t_indices = cos_i2t_sim.min(dim=-1)
            _, min_t2i_indices = cos_t2i_sim.min(dim=-1)

            min_i2t_indices = min_i2t_indices.unsqueeze(1)
            min_t2i_indices = min_t2i_indices.unsqueeze(1)

            ps_images = ps_images[batch_indices, min_t2i_indices.squeeze()]
            ps_caption_ids = ps_caption_ids[batch_indices, min_i2t_indices.squeeze()]
            ps_mask_tokens = ps_mask_tokens[batch_indices, min_i2t_indices.squeeze()]

            num_rows_i2t_replace = int(0.1 * images.size(0))
            rows_i2t_replace = torch.randperm(images.size(0))[:num_rows_i2t_replace]

            num_rows_t2i_replace = int(0.1 * images.size(0))
            rows_t2i_replace = torch.randperm(images.size(0))[:num_rows_t2i_replace]

            images[rows_t2i_replace] = ps_images[rows_t2i_replace]
            caption_ids[rows_i2t_replace] = ps_caption_ids[rows_i2t_replace]
            mask_tokens[rows_i2t_replace] = ps_mask_tokens[rows_i2t_replace]

        image_feats, text_feats = self.base_model(images, caption_ids)
        image_feats, att_img_weights = image_feats
        text_feats, att_txt_weights = text_feats
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})


        if 'sdm' in self.current_task:
            sdm_loss = objectives.compute_sdm(epoch, i_feats, t_feats, batch['pids'], logit_scale,
                                                           args=self.args)
            ret.update({'sdm_loss': 1.0 * sdm_loss})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        with torch.no_grad():
            ids = batch['pids']

            image_norms = i_feats / i_feats.norm(dim=1, keepdim=True)
            text_norms = t_feats / t_feats.norm(dim=1, keepdim=True)

            smi_i2t = image_norms @ text_norms.t()
            smi_t2i = text_norms @ image_norms.t()

            smi_i2t_mod = smi_i2t.clone()
            smi_t2i_mod = smi_t2i.clone()

            ids_eq = ids.view(-1, 1) == ids.view(1, -1)

            smi_i2t_mod[ids_eq] = 0.0
            smi_t2i_mod[ids_eq] = 0.0

            min_i2t_values, min_i2t_indices = smi_i2t_mod.max(dim=1)
            min_t2i_values, min_t2i_indices = smi_t2i_mod.max(dim=1)

        if 'stl_img' in self.current_task:

            text_feats_mean, img_maskeds, sorted_to_indices = [], [], []

            batch_size = image_feats.size(0)

            for i in range(batch_size):
                img = images[i]

                # sorted_indices = sorted_img_indices[i]
                image_feat = image_feats[i][1:, :]
                text_feat = text_feats[i, 0: caption_ids[i].argmax(dim=-1)]
                text_feat_mean = torch.mean(text_feat, dim=0)
                text_feats_mean.append(text_feat_mean)

                similarity = F.cosine_similarity(image_feat, t_feats[i].unsqueeze(0))
                sorted_indices = torch.argsort(similarity, descending=True)

                i_mask_d = int(192 * self.args.i_mask)

                img_masked = self.mask_random_patches_with_noise(img, sorted_indices[192 - i_mask_d: ], patch_size=16,
                                                                 stride_size=16)
                sorted_to_indices.append(sorted_indices[192 - i_mask_d: ])
                img_maskeds.append(img_masked)

            # text_feats_mean = torch.stack(text_feats_mean)
            sorted_to_indices = torch.stack(sorted_to_indices)
            batch_indices = np.arange(batch_size)[:, None]
            local_i_feats = image_feats[:, 1:, :]
            local_i_feats = local_i_feats[batch_indices, sorted_to_indices]

            image_feats_mean = torch.mean(local_i_feats, dim=1)
            img_maskeds = torch.stack(img_maskeds)

            mask_img_feats, _ = self.base_model.encode_image(img_maskeds)

            neg_text_feats = text_feats[min_i2t_indices]

            predicted_img_features = self.imagePre_model(mask_img_feats, text_feats, text_feats)[:, 1:, :]
            neg_predicted_img_features = self.imagePre_model(mask_img_feats, neg_text_feats, neg_text_feats)[:, 1:, :]

            predicted_img_features = predicted_img_features[batch_indices, sorted_to_indices]
            neg_predicted_img_features = neg_predicted_img_features[batch_indices, sorted_to_indices]

            predicted_img_features_mean = torch.mean(predicted_img_features, dim=1)
            neg_predicted_img_features_mean = torch.mean(neg_predicted_img_features, dim=1)


            neg_predicted_img_features_mean = neg_predicted_img_features_mean / neg_predicted_img_features_mean.norm(
                dim=1, keepdim=True)
            predicted_img_features_mean = predicted_img_features_mean / predicted_img_features_mean.norm(dim=1,
                                                                                                         keepdim=True)
            # text_feats_mean = text_feats_mean / text_feats_mean.norm(dim=1, keepdim=True)
            image_feats_mean = image_feats_mean / image_feats_mean.norm(dim=1, keepdim=True)

            neg_image_feats_mean = image_feats_mean[min_t2i_indices]
            # neg_text_feats_mean = text_feats_mean[min_i2t_indices]
            img_pos_dist = F.pairwise_distance(image_feats_mean, predicted_img_features_mean)
            img_neg_dist = F.pairwise_distance(neg_image_feats_mean, predicted_img_features_mean)

            # pre_img_pos_dist = F.pairwise_distance(image_feats_mean, neg_predicted_img_features_mean)
            pre_img_neg_dist = F.pairwise_distance(image_feats_mean, neg_predicted_img_features_mean)

            margin = 0.5
            # text_loss = torch.clamp(text_pos_dist - text_neg_dist + margin, min=0).mean()
            img_loss = torch.clamp(img_pos_dist - img_neg_dist + margin, min=0).mean()
            pre_img_loss = torch.clamp(img_pos_dist - pre_img_neg_dist + margin, min=0).mean()

            pre_loss = img_loss + pre_img_loss

            ret.update({'stl_img_loss': pre_loss})

        if 'stl_txt' in self.current_task:



            with torch.no_grad():
                m_t_feats = self.base_model.encode_text(mask_tokens)[0]

            # m_t_feats = self.base_model.encode_text(mask_tokens)[0]

            txt_maskeds, labels = [], []

            batch_size = caption_ids.size(0)
            with torch.no_grad():
                for i in range(batch_size):
                    mask_token = mask_tokens[i]
                    t_feat = m_t_feats[i]
                    similarity = F.cosine_similarity(t_feat, i_feats[i].unsqueeze(0))
                    sorted_indices = torch.argsort(similarity, descending=False)
                    txt_masked, label = self._build_random_masked_tokens_and_labels(mask_token, sorted_indices)

                    txt_maskeds.append(txt_masked)
                    # sorted_to_indices.append(sort_indice)
                    labels.append(label)

            txt_maskeds = torch.stack(txt_maskeds)
            t_labels = torch.stack(labels)

            mlm_feats, _ = self.base_model.encode_text(txt_maskeds)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = t_labels.reshape(-1)
            pre_txt_loss =  objectives.compute_mlm(scores, mlm_labels)

            # stl_loss =pre_txt_loss + pre_loss

            ret.update({'stl_txt_loss': pre_txt_loss})

        return ret


def build_model(args, num_classes=11003):
    model = FCSA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
