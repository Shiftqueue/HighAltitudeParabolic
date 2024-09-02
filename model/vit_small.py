import torch
import torch.nn.functional as F

from transformers import AutoFeatureExtractor, ViTMSNModel

local_path = "model/models--facebook--vit-msn-small/snapshots/conf/"
print("加载模型，路径", local_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(local_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('使用GPU：', torch.cuda.is_available())
model = ViTMSNModel.from_pretrained(local_path).to(device)


def get_similarity_from_pic(pic1, pic2):
    """
    计算两张图片的余弦相似度
    :param pic1:
    :param pic2:
    :return:
    """
    feature1 = get_pic_feature(pic1)
    feature2 = get_pic_feature(pic2)
    similarity = F.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0))
    return similarity.item()


def get_similarity_from_feature(feature1, feature2):
    """
    计算向量余弦相似度
    :param feature1:
    :param feature2:
    :return:
    """
    similarity = F.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0))
    return similarity.item()


def get_pic_feature(pic):
    """
    提取一张图片的特征向量
    :param pic: 图片
    :return: 384维向量
    """
    inputs = feature_extractor(images=pic, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0][0]
    return last_hidden_states
