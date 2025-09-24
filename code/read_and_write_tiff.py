from osgeo import gdal

def WriteGTiffFile(filename,band, nRows, nCols, data, geotrans, proj, noDataValue, gdalType):  # 向磁盘写入结果文件

    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(filename, nCols, nRows, band, gdalType)
    if(geotrans!="  "):
        ds.SetGeoTransform(geotrans)
    if(proj!="  "):
        ds.SetProjection(proj)
    # outband.FlushCache()
    if(data.ndim==2):
        data=data.reshape(1,data.shape[0],data.shape[1])
    for i in range(1,band+1):
        ds.GetRasterBand(i).SetNoDataValue(noDataValue)
        ds.GetRasterBand(i).WriteArray(data[i-1,:,:])
        ds.GetRasterBand(i).FlushCache()

def read_tiff(inpath):
    ds = gdal.Open(inpath)  # 一般的遥感数据打开方式
    row = ds.RasterYSize  # 获取行数
    col = ds.RasterXSize  # 获取列数
    band = ds.RasterCount  # 获取波段数
    data = ds.ReadAsArray(0, 0, col, row)  # 获取数据
    # print(data.shape)
    # print(data)
    return data



