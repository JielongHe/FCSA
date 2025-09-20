import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cmpm(epoch, image_fetures, text_fetures, pid, logit_scale,  args = None, image_id=None, factor=0.3, epsilon=1e-8):

    # print(image_fetures)
    # print(text_fetures)
    # print(pid)
    # print(logit_scale)
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    local_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    # global_loss = 0.0

    # img_feats, txt_feats = image_fetures.half(), text_fetures.half()
    #
    # # memory_bank.add_features(pid, img_feats, txt_feats)
    #
    # memory_bank.add_features(pid, img_feats, txt_feats)
    # avg_img_features, avg_txt_features = memory_bank.get_average_features(pid, img_feats, txt_feats)
    # avg_image_norm = avg_img_features / avg_img_features.norm(dim=1, keepdim=True)
    # avg_text_norm = avg_txt_features / avg_txt_features.norm(dim=1, keepdim=True)
    #
    # avg_t2i_cosine_theta = avg_text_norm @ avg_image_norm.t()
    # avg_i2t_cosine_theta = avg_t2i_cosine_theta.t()
    #
    # avg_text_proj_image = logit_scale * avg_t2i_cosine_theta
    # avg_image_proj_text = logit_scale * avg_i2t_cosine_theta
    #
    # avg_i2t_pred = F.softmax(avg_image_proj_text, dim=1)
    # avg_i2t_loss = avg_i2t_pred * (
    #         F.log_softmax(avg_image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    # avg_t2i_pred = F.softmax(avg_text_proj_image, dim=1)
    # avg_t2i_loss = avg_t2i_pred * (
    #         F.log_softmax(avg_text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    #
    # global_loss = torch.mean(torch.sum(avg_i2t_loss, dim=1)) + torch.mean(
    #     torch.sum(avg_t2i_loss, dim=1)).float()
    #
    # if args.start <= epoch < args.start+args.d:
    #
    #     memory_bank.add_features(pid, img_feats, txt_feats)
    #
    # if epoch >= args.start+args.d:
    #     memory_bank.add_features(pid, img_feats, txt_feats)
    #     avg_img_features, avg_txt_features = memory_bank.get_average_features(pid, img_feats, txt_feats)
    #     avg_image_norm = avg_img_features / avg_img_features.norm(dim=1, keepdim=True)
    #     avg_text_norm = avg_txt_features / avg_txt_features.norm(dim=1, keepdim=True)
    #
    #     avg_t2i_cosine_theta = avg_text_norm @ avg_image_norm.t()
    #     avg_i2t_cosine_theta = avg_t2i_cosine_theta.t()
    #
    #     avg_text_proj_image = logit_scale * avg_t2i_cosine_theta
    #     avg_image_proj_text = logit_scale * avg_i2t_cosine_theta
    #
    #     avg_i2t_pred = F.softmax(avg_image_proj_text, dim=1)
    #     avg_i2t_loss = avg_i2t_pred * (
    #             F.log_softmax(avg_image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    #     avg_t2i_pred = F.softmax(avg_text_proj_image, dim=1)
    #     avg_t2i_loss = avg_t2i_pred * (
    #             F.log_softmax(avg_text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    #
    #     global_loss = torch.mean(torch.sum(avg_i2t_loss, dim=1)) + torch.mean(
    #         torch.sum(avg_t2i_loss, dim=1)).float()

    # loss = local_loss+global_loss

    return local_loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


