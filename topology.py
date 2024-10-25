import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.cluster import KMeans

def edge_detection(mask):
    """使用Canny边缘检测"""
    edges = feature.canny(mask)
    return edges

def get_key_points(edge, n_points):
    """从边缘提取关键点"""
    y, x = np.where(edge)
    points = np.column_stack((x, y))
    
    if len(points) > n_points:
        kmeans = KMeans(n_clusters=n_points, random_state=0).fit(points)
        key_points = kmeans.cluster_centers_
    else:
        key_points = points
    
    return key_points.astype(int)

def construct_graph(key_points, threshold=20):
    """构建图结构"""
    n = len(key_points)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(key_points[i] - key_points[j])
            if dist < threshold:
                edges.append((i, j))
    return edges

def extract_topology(mask, n_points):
    """提取拓扑结构"""
    edge = edge_detection(mask)
    key_points = get_key_points(edge, n_points)
    graph_edges = construct_graph(key_points)
    return edge, key_points, graph_edges

def visualize_topology(mask, edge, key_points, graph_edges, save_path=None):
    """可视化拓扑结构"""
    plt.figure(figsize=(16, 4))
    
    # 原始掩码
    plt.subplot(141)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Mask')
    plt.axis('off')
    
    # 边缘检测结果
    plt.subplot(142)
    plt.imshow(edge, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    # 关键点
    plt.subplot(143)
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.scatter(key_points[:, 0], key_points[:, 1], c='r', s=10)
    plt.title('Key Points')
    plt.axis('off')
    
    # 拓扑结构
    plt.subplot(144)
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.scatter(key_points[:, 0], key_points[:, 1], c='r', s=10)
    for edge in graph_edges:
        p1, p2 = key_points[edge[0]], key_points[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=0.5)
    plt.title('Topology Graph')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_topology_from_image(image, n_points, save_path=None):
    """从图像提取并可视化拓扑结构"""
    edge, key_points, graph_edges = extract_topology(image, n_points)
    visualize_topology(image, edge, key_points, graph_edges, save_path)

# 示例用法
if __name__ == "__main__":
    # 创建一个示例二值图像
    from glob import glob
    import os
    im_paths = glob('/data/ydchen/VLP/MasaCtrl/图片*.png')
    for image_path in im_paths:
        image = plt.imread(image_path)
        image = np.mean(image, axis=2) > 0.5
        save_path_father = '/data/ydchen/VLP/MasaCtrl/topology/'
        os.makedirs(save_path_father, exist_ok=True)
        save_path = save_path_father + image_path.split('/')[-1]
        # 设置关键点数量
        n_points = 50

        # 提取并可视化拓扑结构
        visualize_topology_from_image(image, n_points, save_path)
