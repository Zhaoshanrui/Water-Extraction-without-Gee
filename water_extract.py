import rasterio
import numpy as np
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
import json
from typing import Tuple, Dict, Any
from rasterio.warp import reproject, Resampling


class WaterExtractionOptimized:
    def __init__(self):
        self.profile = None

    def load_band(self, band_path: str) -> np.ndarray:
        """加载波段数据并转换为浮点型"""
        with rasterio.open(band_path) as src:
            band = src.read(1).astype('float32')
            if self.profile is None:
                self.profile = src.profile.copy()
        return band

    def load_qa_band(self, qa_band_path: str) -> np.ndarray:
        """加载QA波段数据，确保与主波段尺寸一致"""
        with rasterio.open(qa_band_path) as src:
            # 如果已经有profile，确保QA波段与主波段尺寸一致
            if self.profile is not None and (
                    src.height != self.profile['height'] or src.width != self.profile['width']):
                # 需要重采样
                qa_band = np.zeros((self.profile['height'], self.profile['width']), dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, 1),
                    destination=qa_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.profile['transform'],
                    dst_crs=self.profile['crs'],
                    resampling=Resampling.nearest
                )
            else:
                qa_band = src.read(1)
        return qa_band

    def apply_cloud_mask(self, mndwi: np.ndarray, qa_band: np.ndarray) -> np.ndarray:
        """
        应用云掩膜，去除云、云影和雪的影响
        基于Landsat QA_PIXEL波段的位掩码
        """
        # 检查形状是否匹配
        if mndwi.shape != qa_band.shape:
            raise ValueError(f"MNDWI形状{mndwi.shape}与QA波段形状{qa_band.shape}不匹配")

        # 创建云掩膜（位1：云影，位3-4：云置信度，位5：云，位4：雪/冰）
        cloud_mask = np.zeros_like(qa_band, dtype=bool)

        # 提取云影（位1）
        cloud_shadow = (qa_band & 0b0000000000000010) != 0

        # 提取云（位5）
        cloud = (qa_band & 0b0000000000100000) != 0

        # 提取云置信度（位3-4），中高置信度
        cloud_confidence = (qa_band & 0b0000000000001100) >> 2
        medium_high_cloud = cloud_confidence >= 2

        # 提取雪/冰（位4）
        snow_ice = (qa_band & 0b0000000000010000) != 0

        # 合并所有掩膜
        cloud_mask = cloud_shadow | cloud | medium_high_cloud | snow_ice

        # 应用掩膜到MNDWI数据
        mndwi_masked = mndwi.copy()
        mndwi_masked[cloud_mask] = np.nan

        print(f"云掩膜应用完成，遮挡像元比例: {np.sum(cloud_mask) / cloud_mask.size * 100:.2f}%")
        return mndwi_masked

    def calculate_mndwi(self, green_band_path: str, swir_band_path: str) -> np.ndarray:
        """计算MNDWI指数"""
        green = self.load_band(green_band_path)
        swir = self.load_band(swir_band_path)

        # 检查两个波段形状是否一致
        if green.shape != swir.shape:
            raise ValueError(f"绿波段形状{green.shape}与SWIR波段形状{swir.shape}不匹配")

        # 避免除以零，将无效值设为NaN
        np.seterr(divide='ignore', invalid='ignore')
        mndwi = (green - swir) / (green + swir)

        return mndwi

    def find_optimal_threshold(self, mndwi: np.ndarray, method: str = "otsu") -> float:
        """
        确定最佳阈值
        方法可选: "otsu", "triangle", "mean", "median"
        """
        # 去除NaN值
        mndwi_flat = mndwi[~np.isnan(mndwi)].flatten()

        if len(mndwi_flat) == 0:
            raise ValueError("MNDWI数据全为NaN，无法计算阈值。")

        if method == "otsu":
            threshold = filters.threshold_otsu(mndwi_flat)
        elif method == "triangle":
            threshold = filters.threshold_triangle(mndwi_flat)
        elif method == "mean":
            threshold = np.mean(mndwi_flat)
        elif method == "median":
            threshold = np.median(mndwi_flat)
        else:
            raise ValueError(f"不支持的阈值方法: {method}")

        print(f"{method}方法计算的阈值: {threshold:.4f}")
        return threshold

    def create_water_mask(self, mndwi: np.ndarray, threshold: float) -> np.ndarray:
        """根据阈值创建水体掩膜"""
        water_mask = np.where(mndwi > threshold, 1, 0).astype(np.uint8)
        return water_mask

    def refine_water_mask(self, water_mask: np.ndarray,
                          min_size: int = 10,
                          dilation_size: int = 2) -> np.ndarray:
        """
        优化水体掩膜，包括去除小斑点和填充小孔洞
        """
        # 去除小斑点
        cleaned_mask = morphology.remove_small_objects(
            water_mask.astype(bool), min_size=min_size
        ).astype(np.uint8)

        # 填充小孔洞
        filled_mask = morphology.remove_small_holes(
            cleaned_mask.astype(bool), area_threshold=min_size
        ).astype(np.uint8)

        # 使用形态学膨胀连接接近的水体
        if dilation_size > 0:
            selem = morphology.disk(dilation_size)
            refined_mask = morphology.binary_dilation(filled_mask, selem).astype(np.uint8)
        else:
            refined_mask = filled_mask

        # 再次去除可能因膨胀产生的小斑点
        refined_mask = morphology.remove_small_objects(
            refined_mask.astype(bool), min_size=min_size
        ).astype(np.uint8)

        # 计算优化前后的变化
        original_water_pixels = np.sum(water_mask)
        refined_water_pixels = np.sum(refined_mask)
        change_percent = (refined_water_pixels - original_water_pixels) / original_water_pixels * 100

        print(f"水体掩膜优化完成，像元变化: {change_percent:+.2f}%")
        print(f"原始水体像元: {original_water_pixels}, 优化后: {refined_water_pixels}")

        return refined_mask

    def extract_small_water_bodies(self, water_mask: np.ndarray,
                                   mndwi: np.ndarray,
                                   size_threshold: int = 100) -> np.ndarray:
        """
        专门提取细小水体，通过降低阈值和连通区域分析
        """
        # 使用较低的阈值捕捉细小水体
        low_threshold = self.find_optimal_threshold(mndwi, "mean") - 0.1
        potential_water = np.where(mndwi > low_threshold, 1, 0).astype(np.uint8)

        # 标记连通区域
        labeled_water = measure.label(potential_water, connectivity=2)
        regions = measure.regionprops(labeled_water)

        # 创建细小水体掩膜（面积小于阈值）
        small_water_mask = np.zeros_like(water_mask)
        for region in regions:
            if region.area < size_threshold and region.area > 5:  # 忽略极小的噪声
                # 获取该区域的坐标
                for coord in region.coords:
                    small_water_mask[coord[0], coord[1]] = 1

        # 将细小水体合并到主水体掩膜
        combined_mask = np.logical_or(water_mask, small_water_mask).astype(np.uint8)

        small_water_pixels = np.sum(small_water_mask)
        print(f"提取到细小水体像元: {small_water_pixels}")

        return combined_mask

    def calculate_accuracy(self, water_mask: np.ndarray, reference_mask: np.ndarray) -> Dict[str, float]:
        """
        计算水体提取精度
        """
        # 确保两个掩膜形状相同
        if water_mask.shape != reference_mask.shape:
            raise ValueError("水体掩膜和参考掩膜形状不匹配")

        # 计算混淆矩阵
        tp = np.sum((water_mask == 1) & (reference_mask == 1))  # 真阳性
        tn = np.sum((water_mask == 0) & (reference_mask == 0))  # 真阴性
        fp = np.sum((water_mask == 1) & (reference_mask == 0))  # 假阳性
        fn = np.sum((water_mask == 0) & (reference_mask == 1))  # 假阴性

        # 计算精度指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        }

    def save_result(self, data: np.ndarray, output_path: str, dtype: Any, nodata: Any = None) -> None:
        """保存结果到TIFF文件"""
        if self.profile is None:
            raise ValueError("没有可用的地理参考信息")

        # 更新元数据
        profile = self.profile.copy()
        profile.update(
            dtype=dtype,
            count=1,
            compress='lzw',
            nodata=nodata
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
        print(f"结果已保存至: {output_path}")

    def visualize_results(self, mndwi: np.ndarray, water_mask: np.ndarray,
                          output_path: str = "water_extraction_visualization.png") -> None:
        """可视化结果"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # MNDWI指数
        mndwi_display = mndwi.copy()
        mndwi_display[np.isnan(mndwi_display)] = -1  # 为了显示，将NaN设为-1

        im1 = ax1.imshow(mndwi_display, cmap='RdYlBu', vmin=-1, vmax=1)
        ax1.set_title('MNDWI Index')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 水体掩膜
        ax2.imshow(water_mask, cmap='Blues')
        ax2.set_title('Extracted Water Mask')

        # 叠加显示
        from matplotlib.colors import ListedColormap
        overlay_cmap = ListedColormap(['black', 'blue', 'red'])  # 非水体、水体、云/无效区域
        overlay = np.zeros_like(mndwi_display, dtype=int)
        overlay[water_mask == 1] = 1  # 水体
        overlay[np.isnan(mndwi)] = 2  # 云/无效区域

        ax3.imshow(overlay, cmap=overlay_cmap, vmin=0, vmax=2)
        ax3.set_title('Water Mask with Cloud/Invalid Areas')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='Non-water'),
            Patch(facecolor='blue', label='Water'),
            Patch(facecolor='red', label='Cloud/Invalid')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


# 主程序
def main():
    # 初始化水体提取器
    water_extractor = WaterExtractionOptimized()

    # 文件路径
    green_band_path = "E:/桌面/mypaper/changsha/Landsat_used/Landsat8_B1_Mosaic/changsha_landsat8/changsha_B3.tif"
    swir_band_path = "E:/桌面/mypaper/changsha/Landsat_used/Landsat8_B1_Mosaic/changsha_landsat8/changsha_B6.tif"
    qa_band_path = "E:/桌面/mypaper/changsha/Landsat_used/Landsat8_B1_Mosaic/changsha_landsat8/changsha_PIXEL.tif"

    # 输出文件路径
    water_mask_output_path = 'extracted_water_mask_changsha.tif'
    mndwi_output_path = 'mndwi_result_optimized_changsha.tif'
    mndwi_masked_output_path = 'mndwi_masked_result_changsha.tif'

    try:
        # 计算MNDWI
        print("计算MNDWI指数...")
        mndwi = water_extractor.calculate_mndwi(green_band_path, swir_band_path)

        # 加载QA波段并应用云掩膜
        print("应用云掩膜...")
        qa_band = water_extractor.load_qa_band(qa_band_path)
        mndwi_masked = water_extractor.apply_cloud_mask(mndwi, qa_band)

        # 确定最佳阈值
        print("确定最佳阈值...")
        threshold = water_extractor.find_optimal_threshold(mndwi_masked, "otsu")

        # 创建水体掩膜
        print("创建水体掩膜...")
        water_mask = water_extractor.create_water_mask(mndwi_masked, threshold)

        # 优化水体掩膜（去除小斑点和填充孔洞）
        print("优化水体掩膜...")
        refined_water_mask = water_extractor.refine_water_mask(water_mask, min_size=20, dilation_size=1)

        # 提取细小水体
        print("提取细小水体...")
        final_water_mask = water_extractor.extract_small_water_bodies(refined_water_mask, mndwi_masked,
                                                                      size_threshold=50)

        # 保存结果
        print("保存结果...")
        water_extractor.save_result(mndwi, mndwi_output_path, rasterio.float32, -9999)
        water_extractor.save_result(mndwi_masked, mndwi_masked_output_path, rasterio.float32, -9999)
        water_extractor.save_result(final_water_mask, water_mask_output_path, rasterio.uint8, 255)

        # 可视化结果
        print("生成可视化结果...")
        water_extractor.visualize_results(mndwi_masked, final_water_mask)

        print("水体提取完成!")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()