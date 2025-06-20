import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from sam_automatic_mask_generator_onnx import SamAutomaticMaskGeneratorOnnx
import rasterio
import os
from skimage.segmentation import slic




def to_uint8(img):
    ori_shape = img.shape

    img = img.flatten()
    nan_mask = np.isnan(img) | (np.abs(img) > 1e6)
    img = img.astype(np.float32)
    valid_value = img[~nan_mask]
    if len(valid_value) <= 100:
        return np.zeros(ori_shape, dtype=np.uint8)

    _vmin, _vmax = np.percentile(valid_value, [0, 95])
    img[img < _vmin] = _vmin
    img[img > _vmax] = _vmax
    img[nan_mask] = _vmin
    img = (img - _vmin) / ((_vmax - _vmin) + 1e-5)
    img = img.reshape(ori_shape)
    img = (img * 255).astype(np.uint8)

    return img


def gray2rgb(img, cmap=plt.cm.gray, vmin=None, vmax=None):
    """
    Convert a gray-scale image to a RGB image.
    """
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    H, W = img.shape
    nan_mask = np.isnan(img) | (np.abs(img) > 1e7)
    img = img.flatten()

    valid_value = img[~np.isnan(img)]
    if len(valid_value) <= 100:
        return np.zeros((H, W, 3), dtype=np.uint8)
    _vmin, _vmax = np.percentile(valid_value, [2, 98])
    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax
    img[img < vmin] = vmin
    img[img > vmax] = vmax
    img[np.isnan(img)] = vmin
    img = (img - vmin) / ((vmax - vmin) + 1e-5)

    img = img.reshape(H, W)
    img = cmap(img, bytes=True)
    img = img[:, :, :3]
    img[nan_mask] = 0
    return img



def correct_masks(noisy_masks: list[np.ndarray]):
    """
    将加噪后的 masks 转换为互不重叠、覆盖全图的 mask
    """
    N, H, W = len(noisy_masks), noisy_masks[0].shape[0], noisy_masks[0].shape[1]
    final_mask = np.zeros((H, W), dtype=int)  # 每个像素属于哪个类别（从 1 开始）

    for idx in range(N):
        mask = noisy_masks[idx]
        final_mask[mask] = idx + 1  # 类别从 1 开始编号

    # 填补未分配区域（可选）
    final_mask = fill_unassigned_pixels(final_mask)

    # 转换回 one-hot 格式
    corrected_masks = np.zeros((N, H, W), dtype=bool)
    for idx in range(N):
        corrected_masks[idx] = final_mask == (idx + 1)

    return corrected_masks


def fill_unassigned_pixels(final_mask: np.ndarray):
    """
    使用最近邻插值填补未被任何 mask 覆盖的空白区域
    """
    unassigned = final_mask == 0
    if not np.any(unassigned):
        return final_mask

    distances, indices = distance_transform_edt(
        unassigned, return_distances=True, return_indices=True
    )
    nearest_labels = final_mask[tuple(indices[:, unassigned])]
    final_mask[unassigned] = nearest_labels
    return final_mask




def masks_to_binary_array(masks):
    N = len(masks)
    H, W = masks[0]["segmentation"].shape
    binary_masks = np.zeros((N, H, W), dtype=bool)
    for i, mask in enumerate(masks):
        binary_masks[i] = mask["segmentation"]
    return binary_masks


def assign_classes_to_masks(binary_masks, ground_truth):
    """
    binary_masks: (N, H, W) bool array
    ground_truth: (H, W) int array，每个像素代表类别
    返回：(H, W) 的预测图 F
    """
    N, H, W = binary_masks.shape
    F = np.zeros((H, W), dtype=np.int64)

    for idx in range(N):
        mask = binary_masks[idx]
        if not np.any(mask):
            continue
        # 统计 mask 内部的类别直方图
        class_ids, counts = np.unique(ground_truth[mask], return_counts=True)
        dominant_class = class_ids[np.argmax(counts)]
        F[mask] = dominant_class

    return F



def generate_binary_masks_from_segments(segments):
    """
    输入：segments - HxW 的整数标签图
    输出：N x H x W 的 bool 类型二值 mask 数组
    """
    unique_labels = np.unique(segments)
    masks = [(segments == label) for label in unique_labels if label != 0]
    return np.stack(masks)


def merge_masks(masks, image_shape):
    merged_mask = np.zeros(image_shape[:2], dtype=np.int32)
    for idx, mask in enumerate(masks):
        merged_mask[mask["segmentation"]] = idx + 1  # 给每个 mask 分配唯一 ID
    return merged_mask

def merge_masks_np(masks):
    """
    合并多个二值掩码，确保不重叠。
    """
    N, H, W = masks.shape
    merged_mask = np.zeros((H, W), dtype=np.int32)
    for idx in range(N):
        mask = masks[idx]
        merged_mask[mask] = idx + 1  # 给每个 mask 分配唯一 ID
    return merged_mask


def fill_gaps(mask_image, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filled_mask = cv2.dilate(mask_image, kernel, iterations=iterations)
    filled_mask = cv2.erode(filled_mask, kernel, iterations=iterations)
    return filled_mask


def generate_mask(
    image, model: SamAutomaticMaskGeneratorOnnx

):
    
    masks = model.generate(image)
    masks.sort(key=lambda x: np.sum(x["segmentation"]), reverse=True)
    for mask in masks:
        mask["segmentation"] = (
            fill_gaps(mask["segmentation"].astype(np.uint8), 3, 2) > 0
        )
   
    masks = merge_masks(masks, image.shape)
    # plt.imshow(masks, cmap='tab20')
    # plt.show()
    masks = generate_binary_masks_from_segments(masks)
   
    non_zero_masks = masks.sum(axis=0) > 0
    zero_mask = ~non_zero_masks
    slic_mask = slic(
        image,
        n_segments=250,
        compactness=10.0,
        max_num_iter=10,
        sigma=0.5,
        spacing=None,
        convert2lab=None,
        enforce_connectivity=True,
        min_size_factor=0.5,
        max_size_factor=2,
        slic_zero=False,
        start_label=1,
        mask=zero_mask,
        channel_axis=-1,
    )
    slic_mask = generate_binary_masks_from_segments(slic_mask)
    masks = np.concatenate([masks, slic_mask], axis=0)
    masks = remove_small_regions(masks, 2)

    return masks

def remove_small_regions(mask, area_threshold=20):
    masks = []
    idx = 0
    for i in range(mask.shape[0]):
        if np.sum(mask[i]) < area_threshold:
            continue
        m = mask[i].copy().astype(np.uint8)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        m = m.astype(np.int32)
        masks.append(m * (idx + 1))
        idx += 1
    masks = np.array(masks)
    masks = np.sum(masks, axis=0)
    masks = fill_unassigned_pixels(masks)
    masks = generate_binary_masks_from_segments(masks)
    return masks


def sim_mask_us3d():
    mask_generator = SamAutomaticMaskGeneratorOnnx(
        './onnx_models/sam_vit_h_image_encoder.onnx',
        './onnx_models/sam_vit_h_mask_decoder.onnx',
        # points_per_side=48,
    )
    # mask_generator = SamAutomaticMaskGenerator(sam)
    img = cv2.imread("image/demo.jpg")
    masks = generate_mask(img, mask_generator)
    merged = merge_masks_np(masks)
    plt.imshow(merged, cmap='tab20')
    plt.show()
  

if __name__ == "__main__":
   sim_mask_us3d()
