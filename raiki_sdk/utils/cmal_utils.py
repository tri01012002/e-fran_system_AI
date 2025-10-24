import torch
import torch.nn as nn

def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map

def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()

        images[batch_index] = image_tmp

    return images

def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.interpolate(map_tpm2, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.interpolate(map_tpm3, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta

        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear', align_corners=False).squeeze()

        images[batch_index] = image_tmp

    return images

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x