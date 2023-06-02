from typing import List
import cv2
import numpy as np
# from functools import reduce

# from .text_mask_utils import complete_mask_fill, filter_masks, complete_mask
from .text_mask_utils import complete_mask_fill, filter_masks, complete_mask, extract_text_regions
from ..utils import TextBlock, Quadrilateral

async def dispatch(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', verbose: bool = False) -> np.ndarray:
    img_resized = cv2.resize(raw_image, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized[mask_resized > 0] = 255

    bboxes_resized = []
    for region in text_regions:
        for l in region.lines:
            a = Quadrilateral(l, '', 0)
            bboxes_resized.append((a.aabb.x // 2, a.aabb.y // 2, a.aabb.w // 2, a.aabb.h // 2))

    # 用于实现去除误伤
    final_mask = extract_text_regions(mask_resized, bboxes_resized)
    # final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.uint8)
    final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    final_mask[final_mask > 0] = 255

    # 定义结构元素
    kernel_size = int(max(final_mask.shape) * 0.025)  # 选择一个合适的核大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用膨胀操作
    dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)  # 根据需要调整迭代次数

    # 精确覆盖修图地区
    # 将处理后的膨胀mask赋值给final_mask
    final_mask = dilated_mask

    final_mask = cv2.resize(final_mask, (raw_mask.shape[1], raw_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    # raw_mask[final_mask == 0] = 0, 0, 0
    raw_mask[final_mask == 0] = 0

    raw_mask = cv2.resize(raw_mask, (raw_image.shape[1], raw_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    kernel_size = int(max(raw_mask.shape) * 0.018)  # 选择一个合适的核大小 0.008
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用膨胀操作
    raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)  # 根据需要调整迭代次数
    return raw_mask
    
    # 新增 排除气泡 start 添加下面代码后， translator=none 时输出图片气泡内仍有文字，没有删掉任何文字
    
    # 读取图像
    #raw_mask = cv2.imread('mask_final.png', cv2.IMREAD_GRAYSCALE)
    img = raw_image

    # 确保两张图片大小相同
    assert raw_mask.shape == img.shape[:2]

    # 二值化 raw_mask，255 表示白色，0 表示黑色
    _, binary_raw_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

    # 对 raw_mask 进行连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_raw_mask, 8, cv2.CV_32S)

    # 从第1个连通域开始遍历，第0个连通域是背景
    for i in range(1, num_labels):
        # 获取当前连通域的信息
        x, y, w, h, size = stats[i]

        # 获取连通域对应的 img 的区域
        img_region = img[y:y+h, x:x+w]

        # 判断连通域周围是否有非白色点，这里判断的逻辑是只要 RGB 中有一个小于 250 的就认为不是白色
        #if np.any(img_region < 250):
            # 将 raw_mask 中的这个连通域删除，即设置为黑色
        #    labels[labels == i] = 0
        
        # 将上面 if 代码段改为下面这些，可输出结果，只是最终图片空白气泡过多
        length0=np.sum(img_region==0)
        length1=np.sum(img_region>0)
        per=100*(length1/length0)
        if per<100:
            labels[labels==i]=0

    # 更新 raw_mask
    raw_mask[labels > 0] = 255
    raw_mask[labels == 0] = 0
    
    # 新增排除气泡 end
    
    
    return raw_mask

    
    



async def dispatch2(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', verbose: bool = False) -> np.ndarray:
    img_resized = cv2.resize(raw_image, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized[mask_resized > 0] = 255
    bboxes_resized = []
    for region in text_regions:
        for l in region.lines:
            a = Quadrilateral(l, '', 0)
            bboxes_resized.append((a.aabb.x // 2, a.aabb.y // 2, a.aabb.w // 2, a.aabb.h // 2))
    mask_ccs, cc2textline_assignment = filter_masks(mask_resized, bboxes_resized)
    if mask_ccs:
        # mask_filtered = reduce(cv2.bitwise_or, mask_ccs)
        # cv2.imwrite(f'result/mask_filtered.png', mask_filtered)
        #cv2.imwrite(f'result/{task_id}/mask_filtered_img.png', overlay_mask(img_resized_2, mask_filtered))
        if method == 'fit_text':
            final_mask = complete_mask(img_resized, mask_ccs, bboxes_resized, cc2textline_assignment)
        else:
            final_mask = complete_mask_fill(img_resized, mask_ccs, bboxes_resized, cc2textline_assignment)
        #cv2.imwrite(f'result/{task_id}/mask.png', final_mask)
        #cv2.imwrite(f'result/{task_id}/mask_img.png', overlay_mask(img_resized_2, final_mask))
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255
    else:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    return final_mask
