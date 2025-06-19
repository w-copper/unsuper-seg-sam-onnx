# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple
import onnxruntime as ort
from copy import deepcopy
from itertools import product
import math
# from segment_anything

class MaskDataOnnx:
    """
    A structure for storing masks and their related data in batched format.
    ONNX-compatible version without torch dependencies.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray)
            ), "MaskDataOnnx only supports list and numpy arrays."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray)
        ), "MaskDataOnnx only supports list and numpy arrays."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self):
        return self._stats.items()

    def filter(self, keep: np.ndarray) -> None:
        """Filter data based on boolean mask."""
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep]
            elif isinstance(v, list) and keep.dtype == bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskDataOnnx key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskDataOnnx") -> None:
        """Concatenate with another MaskDataOnnx."""
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskDataOnnx key {k} has an unsupported type {type(v)}.")


class ResizeLongestSideOnnx:
    """
    ONNX-compatible version of ResizeLongestSide transform.
    """
    
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return cv2.resize(image, (target_size[1], target_size[0]))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class SamPredictorOnnx:
    """
    ONNX-compatible version of SamPredictor.
    """
    
    def __init__(
        self,
        image_encoder_path: str,
        mask_decoder_path: str,
        img_size: int = 1024,
        mask_threshold: float = 0.0,
        image_format: str = "RGB",
    ) -> None:
        """
        Uses ONNX models to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          image_encoder_path (str): Path to the image encoder ONNX model.
          mask_decoder_path (str): Path to the mask decoder ONNX model.
          img_size (int): The size the image encoder expects.
          mask_threshold (float): Threshold for mask binarization.
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.image_encoder = ort.InferenceSession(image_encoder_path, providers=['CUDAExecutionProvider'])
        self.mask_decoder = ort.InferenceSession(mask_decoder_path, providers=['CUDAExecutionProvider'])
        self.img_size = img_size
        self.mask_threshold = mask_threshold
        self.image_format = image_format
        self.transform = ResizeLongestSideOnnx(img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image = self.preprocess(input_image)
        
        self.set_image_embedding(input_image, image.shape[:2])

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        x = x.astype(np.float32)
        # Pad
        h, w = x.shape[:2]
        padh = self.img_size - h
        padw = self.img_size - w
        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant')
        
        # Convert to BCHW format
        x = x.transpose(2, 0, 1)[None, :, :, :]
        return x

    def set_image_embedding(
        self,
        transformed_image: np.ndarray,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image.
        """
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = transformed_image.shape[-2:]
        
        # Run image encoder
        self.features = self.image_encoder.run(None, {"images": transformed_image})[0]
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        mask_per_points: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_np, labels_np = None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            if mask_per_points:
                coords_np, labels_np = point_coords[ : ,None, :], point_labels[ :, None]
            else:
                coords_np, labels_np = point_coords[None, :, :], point_labels[None, :]
        if box is not None:
            box = self.transform.apply_coords(box.reshape(-1, 2), self.original_size)
            box_coords = box.reshape(2, 2)
            box_labels = np.array([2, 3])
            if coords_np is not None:
                coords_np = np.concatenate([coords_np, box_coords[None, :, :]], axis=1)
                labels_np = np.concatenate([labels_np, box_labels[None, :]], axis=1)
            else:
                coords_np, labels_np = box_coords[None, :, :], box_labels[None, :]

        # Prepare inputs for mask decoder
        if coords_np is None:
            coords_np = np.zeros((1, 1, 2), dtype=np.float32)
            labels_np = np.array([[-1]], dtype=np.float32)
        else:
            coords_np = coords_np.astype(np.float32)
            labels_np = labels_np.astype(np.float32)

        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([[0]], dtype=np.float32)
        else:
            mask_input = mask_input[None, :, :, :]
            has_mask_input = np.array([[1]], dtype=np.float32)
        # print(coords_np.shape)
        # Run mask decoder
        onnx_inputs = {
            "image_embeddings": self.features,
            "point_coords": coords_np,
            "point_labels": labels_np,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(self.original_size, dtype=np.float32),
        }
        masks, iou_predictions, low_res_masks = self.mask_decoder.run(None, onnx_inputs)
        # print(masks.shape)
        
        # Select the correct number of masks
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_predictions = iou_predictions[:, mask_slice]
        low_res_masks = low_res_masks[:, mask_slice, :, :]

        if not return_logits:
            masks = masks > self.mask_threshold
        if mask_per_points:
            return masks, iou_predictions, low_res_masks
        return masks[0], iou_predictions[0], low_res_masks[0]

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = n_per_side // (scale_per_layer**i)
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """Generates a list of crop boxes of different sizes."""
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(im_w, x0 + crop_w), min(im_h, y0 + crop_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def batch_iterator(batch_size: int, *args):
    """Yields batches of the given arguments."""
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: np.ndarray, mask_threshold: float, threshold_offset: float) -> np.ndarray:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        & (masks > (mask_threshold - threshold_offset))
    ).sum(axis=(-1, -2), dtype=np.float32)
    unions = (
        (masks > (mask_threshold + threshold_offset))
        | (masks > (mask_threshold - threshold_offset))
    ).sum(axis=(-1, -2), dtype=np.float32)
    return intersections / unions


def batched_mask_to_box(masks: np.ndarray) -> np.ndarray:
    """
    Calculates boxes in XYXY format around the given masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # Flatten to (..., H, W)
    flat_masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    boxes = np.zeros((flat_masks.shape[0], 4), dtype=np.float32)
    
    for i, mask in enumerate(flat_masks):
        if mask.sum() == 0:
            continue
        y_indices, x_indices = np.where(mask > 0)
        boxes[i] = np.array([x_indices.min(), y_indices.min(), x_indices.max() + 1, y_indices.max() + 1])
    
    return boxes.reshape(masks.shape[:-2] + (4,))


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Computes the area of a set of bounding boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        # Compute IoU with remaining boxes
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Remove boxes with high IoU
        order = order[1:][ious <= iou_threshold]
    
    return np.array(keep, dtype=np.int64)


def mask_to_rle_numpy(tensor: np.ndarray) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.transpose(0, 2, 1).reshape(b, -1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[0] == i
        cur_idxs = change_indices[1][cur_idxs]
        if len(cur_idxs) % 2 == 1:
            cur_idxs = np.concatenate([cur_idxs, np.array([tensor.shape[-1] - 1])])
        cur_idxs = cur_idxs.reshape(-1, 2)
        cur_idxs[:, 1] += 1
        lengths = cur_idxs[:, 1] - cur_idxs[:, 0]
        out.append({"size": [h, w], "counts": lengths.tolist()})

    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.zeros(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def area_from_rle(rle: Dict[str, Any]) -> int:
    """Compute the area of a mask from its RLE."""
    return sum(rle["counts"][1::2])


def uncrop_boxes_xyxy(boxes: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """Uncrop boxes from a crop back to the original image."""
    x0, y0, _, _ = crop_box
    offset = np.array([x0, y0, x0, y0])
    # Check if boxes has the right shape
    if len(boxes.shape) == 1:
        offset = offset.reshape(-1)
    else:
        offset = offset.reshape(1, -1)
    return boxes + offset


def uncrop_points(points: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """Uncrop points from a crop back to the original image."""
    x0, y0, _, _ = crop_box
    offset = np.array([x0, y0])
    # Check if points has the right shape
    if len(points.shape) == 1:
        offset = offset.reshape(-1)
    else:
        offset = offset.reshape(1, -1)
    return points + offset


def uncrop_masks(masks: np.ndarray, crop_box: List[int], orig_h: int, orig_w: int) -> np.ndarray:
    """Uncrop masks from a crop back to the original image."""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad_width = ((0, 0), (y0, pad_y - y0), (x0, pad_x - x0))
    return np.pad(masks, pad_width, mode="constant", constant_values=0)


def remove_small_regions(mask: np.ndarray, area_thresh: int, mode: str) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2
    
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def is_box_near_crop_edge(
    boxes: np.ndarray, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> np.ndarray:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_np = np.array(crop_box, dtype=np.float32)
    orig_box_np = np.array(orig_box, dtype=np.float32)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).astype(np.float32)
    near_crop_edge = np.abs(boxes - crop_box_np[None, :]) < atol
    near_image_edge = np.abs(boxes - orig_box_np[None, :]) < atol
    near_crop_edge = near_crop_edge & ~near_image_edge
    return np.any(near_crop_edge, axis=1)


def box_xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    """Convert box from xyxy to xywh format."""
    box_xywh = box_xyxy.copy()
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


class SamAutomaticMaskGeneratorOnnx:
    """
    ONNX-compatible version of SamAutomaticMaskGenerator.
    Uses ONNX runtime for inference instead of PyTorch.
    """
    
    def __init__(
        self,
        image_encoder_path: str,
        mask_decoder_path: str,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        img_size: int = 1024,
        mask_threshold: float = 0.0,
    ) -> None:
        """
        Using ONNX SAM models, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks.

        Arguments:
          image_encoder_path (str): Path to the image encoder ONNX model.
          mask_decoder_path (str): Path to the mask decoder ONNX model.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          img_size (int): The size the image encoder expects.
          mask_threshold (float): Threshold for mask binarization.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."

        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictorOnnx(
            image_encoder_path=image_encoder_path,
            mask_decoder_path=mask_decoder_path,
            img_size=img_size,
            mask_threshold=mask_threshold,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            from pycocotools import mask as mask_utils
            mask_data["segmentations"] = [mask_utils.encode(np.asfortranarray(rle_to_mask(rle).astype(np.uint8))) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = mask_data['masks']
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": float(mask_data["iou_preds"][idx]),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": float(mask_data["stability_score"][idx]),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskDataOnnx:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskDataOnnx()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            keep_by_nms = nms(
                data["boxes"].astype(np.float32),
                scores,
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskDataOnnx:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskDataOnnx()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = nms(
            data["boxes"].astype(np.float32),
            data["iou_preds"],
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = np.array([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskDataOnnx:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = transformed_points.astype(np.float32)
        in_labels = np.ones(in_points.shape[0], dtype=np.float32)
       
        # print(labels)
        masks, iou_preds, _ = self.predictor.predict(
                point_coords=in_points,
                point_labels=in_labels,
                multimask_output=True,
                return_logits=True,
                mask_per_points=True
            )
      
        
        # max_id = np.argmax(iou_preds, axis=1)
        # masks = masks[np.arange(iou_preds.shape[0]),max_id]
        # iou_preds = iou_preds[np.arange(iou_preds.shape[0]),max_id]
        # print(masks.shape)
        # print(iou_preds.shape)
        # Serialize predictions and store in MaskDataOnnx
        data = MaskDataOnnx(
            masks=masks.reshape(-1, *masks.shape[2:]),
            iou_preds=iou_preds.flatten(),
            points=np.repeat(points, masks.shape[1], axis=0),
        )
        
        # print(data['masks'].shape)
        # print(data['iou_preds'].shape)
        # print(data['points'].shape)
 
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not np.all(keep_mask):
            data.filter(keep_mask)
        if np.sum(keep_mask) == 0:
            return data
        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
    
        data["rles"] = mask_to_rle_numpy(data["masks"])
        
        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskDataOnnx, min_area: int, nms_thresh: float
    ) -> MaskDataOnnx:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires opencv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(mask[None, :, :])
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = np.concatenate(new_masks, axis=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = nms(
            boxes.astype(np.float32),
            np.array(scores),
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_np = masks[i_mask:i_mask+1]
                mask_data["rles"][i_mask] = mask_to_rle_numpy(mask_np)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Automatic Mask Generator ONNX")
    parser.add_argument("--image_encoder", required=True, help="Path to image encoder ONNX model")
    parser.add_argument("--mask_decoder", required=True, help="Path to mask decoder ONNX model")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--points_per_side", type=int, default=32, help="Points per side")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="Prediction IoU threshold")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95, help="Stability score threshold")
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask generator
    mask_generator = SamAutomaticMaskGeneratorOnnx(
        image_encoder_path=args.image_encoder,
        mask_decoder_path=args.mask_decoder,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    
    # Save results
    import os
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save mask annotations
    with open(os.path.join(args.output_dir, "masks.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        masks_serializable = []
        for mask in masks:
            mask_copy = mask.copy()
            if isinstance(mask_copy["segmentation"], np.ndarray):
                mask_copy["segmentation"] = mask_copy["segmentation"].tolist()
            masks_serializable.append(mask_copy)
        json.dump(masks_serializable, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")