# P02: image batch shape, reshape, and axis operations
import numpy as np

def image_batch_summary(images):
    if images.ndim != 3:
        return None
    result = {"original_shape": images.shape, "flat": None, "pixel_sums": None, "brightest_indices": None}
    result["flat"] = flat = images.reshape(images.shape[0],-1)
    result["pixel_sums"] = flat.sum(axis=1)
    result["brightest_indices"] = flat.argmax(axis=1)
    return result

if __name__ == "__main__":
    from test_helpers import check_array_equal, check_equal, check_is_none
    images = np.array([
        [[1, 2], [3, 4]],
        [[9, 0], [2, 1]],
    ])
    summary = image_batch_summary(images)
    check_equal("P02/保留原始shape", summary["original_shape"], (2, 2, 2))
    check_array_equal(
        "P02/展平图像",
        summary["flat"],
        np.array([[1, 2, 3, 4], [9, 0, 2, 1]]),
    )
    check_array_equal("P02/逐图像像素和", summary["pixel_sums"], np.array([10, 12]))
    check_array_equal("P02/最亮像素索引", summary["brightest_indices"], np.array([3, 0]))
    check_is_none("P02/拒绝非三维输入", image_batch_summary(np.array([1, 2, 3])))