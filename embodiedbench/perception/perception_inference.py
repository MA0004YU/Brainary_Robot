import torch
import clip
from PIL import Image
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# -------------------------- 1. 初始化CLIP模型 --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------- 2. 可配置参数 --------------------------
# 相似度阈值（可根据场景调整）
SIM_THRESHOLD_SYNONYM = 0.85    # 同义关系阈值（高相似度）
SIM_THRESHOLD_ANTONYM = 0.2     # 反义关系阈值（低相似度）
SIM_THRESHOLD_METAPHOR = 0.6    # 比喻关系阈值（中等相似度+领域差异）
CLUSTER_THRESHOLD_HIERARCHY = 0.5  # 层次关联聚类阈值
CLUSTER_THRESHOLD_DOMAIN = 0.4     # 领域聚类阈值（越小领域划分越细）

# -------------------------- 3. 核心工具函数 --------------------------
def get_clip_features(texts=None, images=None):
    """
    利用CLIP提取文本/图像特征（归一化）
    :param texts: 文本列表
    :param images: 图像路径列表
    :return: 特征张量（归一化后）、标签列表（文本/图像路径）
    """
    features = None
    labels = []
    if texts is not None and len(texts) > 0:
        # 文本编码
        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            features = model.encode_text(text_tokens)
        labels = texts.copy()
    elif images is not None and len(images) > 0:
        # 图像编码
        image_tensors = []
        valid_images = []
        for img_path in images:
            try:
                img = Image.open(img_path)
                image_tensors.append(preprocess(img).unsqueeze(0).to(device))
                valid_images.append(img_path)
            except:
                print(f"警告：图像 {img_path} 无法读取，已跳过")
        if len(image_tensors) == 0:
            return None, []
        image_batch = torch.cat(image_tensors)
        with torch.no_grad():
            features = model.encode_image(image_batch)
        labels = valid_images.copy()
    
    # 特征归一化（余弦相似度计算前提）
    if features is not None:
        features = features / features.norm(dim=-1, keepdim=True)
    return features, labels

def cosine_similarity_matrix(features):
    """计算特征矩阵的余弦相似度矩阵（n×n）"""
    if features is None:
        return None
    sim_matrix = torch.matmul(features, features.T).cpu().numpy()
    # 处理数值精度问题（避免相似度>1或<0）
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    return sim_matrix

def auto_domain_clustering(texts):
    """
    全自动文本领域聚类（无监督）
    :param texts: 待聚类的文本列表
    :return: 文本→领域ID的映射、领域ID→文本列表的映射
    """
    # 提取文本特征
    text_feats, text_labels = get_clip_features(texts=texts)
    if text_feats is None or len(text_labels) < 2:
        # 文本数量不足时，所有文本归为同一领域
        text2domain = {t: 0 for t in texts}
        domain2texts = {0: texts}
        return text2domain, domain2texts
    
    # 计算距离矩阵（距离=1-相似度）
    sim_matrix = cosine_similarity_matrix(text_feats)
    dist_matrix = 1 - sim_matrix
    # 层级聚类（ward算法：最小化类内方差）
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    # 按阈值划分领域聚类
    domain_labels = fcluster(linkage_matrix, t=CLUSTER_THRESHOLD_DOMAIN, criterion='distance')
    
    # 构建映射关系
    text2domain = {}
    domain2texts = {}
    for text, domain_id in zip(text_labels, domain_labels):
        text2domain[text] = domain_id
        if domain_id not in domain2texts:
            domain2texts[domain_id] = []
        domain2texts[domain_id].append(text)
    
    # 打印自动划分的领域结果（便于调试）
    print("\n===== 自动划分的文本领域 =====")
    for domain_id, texts_in_domain in domain2texts.items():
        print(f"领域ID {domain_id}：{texts_in_domain}")
    
    return text2domain, domain2texts

# -------------------------- 4. 自动关联挖掘函数 --------------------------
def auto_linguistic_relation(texts):
    """
    自动挖掘语言学关联（同义/反义/比喻）
    逻辑：
    - 同义：相似度 > SIM_THRESHOLD_SYNONYM + 同自动聚类领域
    - 反义：相似度 < SIM_THRESHOLD_ANTONYM + 同自动聚类领域
    - 比喻：SIM_THRESHOLD_METAPHOR < 相似度 < SIM_THRESHOLD_SYNONYM + 不同自动聚类领域
    """
    print("\n===== 自动挖掘语言学关联（同义/反义/比喻） =====")
    # 第一步：全自动领域聚类
    text2domain, _ = auto_domain_clustering(texts)
    
    # 第二步：提取文本特征并计算相似度
    text_feats, text_labels = get_clip_features(texts=texts)
    if text_feats is None:
        return {}
    sim_matrix = cosine_similarity_matrix(text_feats)
    n = len(text_labels)
    
    linguistic_rels = {
        "同义": [], "反义": [], "比喻": []
    }
    
    # 遍历所有文本对（避免重复计算，i<j）
    for i in range(n):
        for j in range(i+1, n):
            text1 = text_labels[i]
            text2 = text_labels[j]
            sim = round(sim_matrix[i][j], 4)
            # 自动判断是否同领域（通过聚类ID）
            domain1 = text2domain[text1]
            domain2 = text2domain[text2]
            same_domain = (domain1 == domain2)
            
            # 判断关联类型
            if sim >= SIM_THRESHOLD_SYNONYM and same_domain:
                rel_type = "同义"
                linguistic_rels[rel_type].append((text1, text2, sim))
                print(f"[{text1}] ↔ [{text2}] | 同义 | 相似度：{sim} | 领域ID：{domain1}")
            elif sim <= SIM_THRESHOLD_ANTONYM and same_domain:
                rel_type = "反义"
                linguistic_rels[rel_type].append((text1, text2, sim))
                print(f"[{text1}] ↔ [{text2}] | 反义 | 相似度：{sim} | 领域ID：{domain1}")
            elif SIM_THRESHOLD_METAPHOR <= sim < SIM_THRESHOLD_SYNONYM and not same_domain:
                rel_type = "比喻"
                linguistic_rels[rel_type].append((text1, text2, sim))
                print(f"[{text1}] ↔ [{text2}] | 比喻 | 相似度：{sim} | 领域ID：{domain1}/{domain2}")
    
    return linguistic_rels

def auto_hierarchy_relation(texts, plot_dendrogram=False):
    """
    自动挖掘层次关联（跨层次/同层次）
    逻辑：
    - 层级聚类生成树状结构 → 按阈值划分聚类层级
    - 跨层次：父聚类 → 子聚类（不同层级）
    - 同层次：同一聚类内的节点（兄弟节点）
    """
    print("\n===== 自动挖掘层次关联（跨层次/同层次） =====")
    # 提取文本特征
    text_feats, text_labels = get_clip_features(texts=texts)
    if text_feats is None:
        return {}
    # 计算相似度矩阵 → 转换为距离矩阵（距离=1-相似度）
    sim_matrix = cosine_similarity_matrix(text_feats)
    dist_matrix = 1 - sim_matrix
    # 层级聚类（ward算法：最小化类内方差）
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    
    # 可选：绘制树状图（直观展示层级）
    if plot_dendrogram:
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix, labels=text_labels)
        plt.title("文本层级聚类树状图")
        plt.xlabel("文本")
        plt.ylabel("距离")
        plt.show()
    
    # 按阈值划分聚类（得到每个文本的聚类标签）
    cluster_labels = fcluster(linkage_matrix, t=CLUSTER_THRESHOLD_HIERARCHY, criterion='distance')
    # 构建聚类→文本的映射
    cluster2texts = {}
    for idx, (text, cid) in enumerate(zip(text_labels, cluster_labels)):
        if cid not in cluster2texts:
            cluster2texts[cid] = []
        cluster2texts[cid].append(text)
    
    # 1. 同层次关联：同一聚类内的文本对
    same_level_rels = []
    for cid, texts_in_cluster in cluster2texts.items():
        if len(texts_in_cluster) < 2:
            continue
        # 生成聚类内所有文本对
        for i in range(len(texts_in_cluster)):
            for j in range(i+1, len(texts_in_cluster)):
                text1 = texts_in_cluster[i]
                text2 = texts_in_cluster[j]
                sim = round(sim_matrix[text_labels.index(text1)][text_labels.index(text2)], 4)
                same_level_rels.append((text1, text2, sim))
                print(f"同层次 | [{text1}] ↔ [{text2}] | 相似度：{sim} | 聚类ID：{cid}")
    
    # 2. 跨层次关联：基于聚类树的父子层级（简化版：距离近的大类包含小类）
    cross_level_rels = []
    # 按聚类大小排序（大类在前）
    sorted_clusters = sorted(cluster2texts.items(), key=lambda x: len(x[1]), reverse=True)
    # 匹配大类包含的小类（简化逻辑，可根据需求优化）
    for i in range(len(sorted_clusters)):
        big_cid, big_texts = sorted_clusters[i]
        for j in range(i+1, len(sorted_clusters)):
            small_cid, small_texts = sorted_clusters[j]
            # 计算大类和小类的平均相似度
            big_feats = text_feats[[text_labels.index(t) for t in big_texts]]
            small_feats = text_feats[[text_labels.index(t) for t in small_texts]]
            avg_sim = round(torch.matmul(big_feats.mean(dim=0, keepdim=True), 
                                        small_feats.mean(dim=0, keepdim=True).T).cpu().numpy()[0][0], 4)
            if avg_sim > SIM_THRESHOLD_METAPHOR:  # 相似度足够则判定为跨层次
                cross_level_rels.append((big_texts[0], small_texts[0], avg_sim))  # 简化取代表词
                print(f"跨层次 | [{big_texts[0]}] → [{small_texts[0]}] | 平均相似度：{avg_sim} | 大类{big_cid}→小类{small_cid}")
    
    hierarchy_rels = {
        "同层次": same_level_rels,
        "跨层次": cross_level_rels
    }
    return hierarchy_rels

def auto_cross_modal_relation(image_paths, texts):
    """自动挖掘跨模态关联（图像-文本）"""
    print("\n===== 自动挖掘跨模态关联（图像-文本） =====")
    # 提取特征
    img_feats, img_labels = get_clip_features(images=image_paths)
    text_feats, text_labels = get_clip_features(texts=texts)
    if img_feats is None or text_feats is None:
        return {}
    
    # 计算图像-文本相似度矩阵
    sim_matrix = torch.matmul(img_feats, text_feats.T).cpu().numpy()
    cross_modal_rels = []
    
    # 输出所有图像-文本对的关联
    for i, img_path in enumerate(img_labels):
        for j, text in enumerate(text_labels):
            sim = round(sim_matrix[i][j], 4)
            cross_modal_rels.append((img_path, text, sim))
            print(f"图像[{img_path}] ↔ 文本[{text}] | 相似度：{sim}")
    
    return cross_modal_rels

# -------------------------- 5. 主推理流程 --------------------------
def main():
    # 示例输入（可替换为任意图像/文本，无需预定义领域）
    image_paths = [
        "house.jpg",    # 房屋实景图
        "kitchen.jpg",  # 厨房图
        "sun.jpg"       # 太阳图
    ]
    text_queries = [
        "太阳", "月亮", "火球", "日头",  # 语言学关联候选
        "房屋", "厨房", "卧室", "刀具", "碗筷"  # 层次关联候选
    ]
    
    # 1. 自动挖掘跨模态关联
    cross_modal_rels = auto_cross_modal_relation(image_paths, text_queries)
    
    # 2. 自动挖掘语言学关联（含全自动领域判断）
    linguistic_rels = auto_linguistic_relation(text_queries)
    
    # 3. 自动挖掘层次关联（plot_dendrogram=True 可绘制聚类树）
    hierarchy_rels = auto_hierarchy_relation(text_queries, plot_dendrogram=False)

if __name__ == "__main__":
    main()
