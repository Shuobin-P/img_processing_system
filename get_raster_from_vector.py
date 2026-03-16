import geopandas as gpd
import rasterio
import pandas as pd

# 读取矢量
gdf = gpd.read_file("D:\Data\KADE-XGBoost\Training_Test_Data\CH_AGB\Plot_level_CH_AGB.shp")
print("gdf= ")
print(gdf["geometry"])
# 只保留 Location 为 Tahura Ngurah Rai 的 Polygon
gdf = gdf[(gdf["Location"] == "Tahura Ngurah Rai") & (gdf["Type"] == "Under restoration")]
# 打开栅格
with rasterio.open("D:\Data\KADE-XGBoost\Training_Test_Data\Tahura_Ngurah_Rai\B1-B7_Vegetation_Texture_BandInfo_Layer_Stack_Result-Based_on_pca_b1-b7.tif") as src:
    coords = [(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry]
    values = list(src.sample(coords)) 

# 转为 DataFrame
df = pd.DataFrame(values, columns=[f'B{i+1}' for i in range(src.count)])

# 加入矢量属性
df["AGB"] = gdf["AGB_Chave"].values # 顺序正确

df.to_csv("D:\Data\KADE-XGBoost\Training_Test_Data\Tahura_Ngurah_Rai\All_bands_AGB_based_on_b1-b7_pca.csv", index=False)


