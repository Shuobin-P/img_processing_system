from osgeo import gdal, ogr

# -------------------------
# 输入文件
# -------------------------
raster_path = "D:\Data\KADE-XGBoost\Training_Test_Data\Tahura_Ngurah_Rai\Tahura_Ngurah_Rai_15m.tif"
vector_path = "D:\Data\KADE-XGBoost\Training_Test_Data\CH_AGB\Plot_level_CH_AGB.shp"
output_csv = "D:\Data\KADE-XGBoost\Training_Test_Data\Tahura_Ngurah_Rai\output.csv"
ds = ogr.Open(vector_path, 0)
print("LayerCount= ", ds.GetLayerCount())
lyr = ds.GetLayer(0)
layer_defn = lyr.GetLayerDefn()
field_names = [layer_defn.GetFieldDefn(i).GetName()
               for i in range(layer_defn.GetFieldCount())]
print("field_names= ", field_names)
counter = 0
for feat in lyr:
    #pt = feat.geometry()
    #x = pt.GetX()
    #y = pt.GetY()
    location = feat.GetField('Location')
    agb = feat.GetField('AGB_Chave')
    print(location, agb)
    break
del ds