import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
from einops import rearrange

# add
import clip
import math
from model.get_cam import get_img_cam
from pytorch_grad_cam import GradCAM
from clip.clip_text import new_class_names, new_class_names_coco
import matplotlib.pyplot as plt
import model.resnet as resnet

def zeroshot_classifier(classnames, templates, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm
       
        backbone = resnet.__dict__['resnet50'](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.backbone.eval()

        fea_dim = 1024 + 512

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        if self.shot==1:
            channel = 516
        else:
            channel = 524
        self.query_merge = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformer(shot=self.shot)

        # add
        self.annotation_root = args.annotation_root
        self.clip_model, _ = clip.load(args.clip_path)
        if self.dataset == 'pascal':
            self.bg_text_features = zeroshot_classifier(new_class_names, ['a photo without {}.'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names, ['a photo of {}.'],
                                                        self.clip_model)
        elif self.dataset == 'coco':
            self.bg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo without {}.'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo of {}.'],
                                                        self.clip_model)

    def forward(self, x, x_cv2, que_name, class_name, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        h, w = x.shape[-2:]

        _, _, query_feat_2, query_feat_3 = self.extract_feats(x)
        query_feat_cnn = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_cnn = self.down_query(query_feat_cnn)

        target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
        cam = GradCAM(model=self.clip_model, target_layers=target_layers, reshape_transform=reshape_transform)
        img_cam_list, img_cam_list2 = get_img_cam(x_cv2, que_name, class_name, self.clip_model, self.bg_text_features, self.fg_text_features, cam, self.annotation_root, self.training)

        img_cam_list = [F.interpolate(t_img_cam.unsqueeze(0).unsqueeze(0), size=(query_feat_cnn.shape[2], query_feat_cnn.shape[3]), mode='bilinear',
                                      align_corners=True) for t_img_cam in img_cam_list]
        img_cam = torch.cat(img_cam_list, 0)

        img_cam_list2 = [F.interpolate(t_img_cam.unsqueeze(0).unsqueeze(0), size=(query_feat_cnn.shape[2], query_feat_cnn.shape[3]), mode='bilinear',
                                      align_corners=True) for t_img_cam in img_cam_list2]
        img_cam2 = torch.cat(img_cam_list2, 0)

        img_cam = img_cam.repeat(1,2,1,1)
        img_cam2 = img_cam2.repeat(1,2,1,1)
        img_mask = img_cam[:,1,:,:].unsqueeze(1)
        img_mask2 = img_cam2[:,1,:,:].unsqueeze(1)

        query_pro1 = Weighted_GAP(query_feat_cnn, \
                                F.interpolate(img_mask, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        query_pro2 = Weighted_GAP(query_feat_cnn, \
                                 F.interpolate(img_mask2, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
                                               mode='bilinear', align_corners=True))
        query_pro = torch.mean(torch.cat([query_pro1.unsqueeze(0), query_pro2.unsqueeze(0)], 0),dim=0)

        query_feat_bin = query_pro.repeat(1, 1, query_feat_cnn.shape[-2], query_feat_cnn.shape[-1])

        query_feat0 = self.supp_merge(torch.cat([query_feat_cnn, query_feat_bin],
                                              dim=1))

        bs = x.shape[0]

        query_feat_bin = rearrange(query_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        query_feat_bin = torch.mean(query_feat_bin, dim=1)
        query_feat = self.query_merge(torch.cat([query_feat_cnn, query_feat_bin, img_cam * 10, img_cam * 10], dim=1))

        meta_out, weights = self.transformer(query_feat, query_feat0, img_mask, img_cam, img_cam2)

        # Output Part
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            aux_loss1 = self.criterion(meta_out, y_m.long())

            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return meta_out.max(1)[1], aux_loss1, distil_loss / 3, distil_loss / 3
        else:
            return meta_out, meta_out, meta_out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.transformer.mix_transformer.parameters()},
                {'params': model.supp_merge.parameters(), "lr": LR * 10},
                {'params': model.query_merge.parameters(), "lr": LR * 10},
                {'params': model.down_supp.parameters(), "lr": LR * 10},
                {'params': model.down_query.parameters(), "lr": LR * 10},

            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer
   
    def freeze_modules(self, model):
        for param in model.backbone.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
        return results
