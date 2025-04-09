import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ee
import geemap
import random
import json
import time

import VQ_VAE

class TSClusterer:

  def __init__(self, aoi, startYear, endYear, cloudTH=5, percentReductionTH=20, verbose=True):
    
    ndviList = []
    rgbImgList = []

    if verbose:
      print('## Image availability by year.')

    for i1 in range(startYear,endYear+1,1):

      LSCol = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate(''+str(i1)+'-01-01', ''+str(i1)+'-12-31').filterBounds(aoi)
      LSCol = LSCol.filterMetadata('CLOUD_COVER','less_than',cloudTH)

      if verbose:
        if LSCol.size().getInfo()==0:
          print('Image for the year ' + str(i1) + ' not found.')
          continue
        else:
          print('Image for the year ' + str(i1) + ' found.')
      
      def cloudMask(img):
        qaMask = img.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturationMask = img.select('QA_RADSAT').eq(0)
        return img.select(['SR_B2','SR_B3','SR_B4','SR_B5']).updateMask(qaMask).updateMask(saturationMask)
      LSCol = LSCol.map(cloudMask)

      LSImg = LSCol.select(['SR_B2','SR_B3','SR_B4','SR_B5']).reduce(ee.Reducer.percentile([percentReductionTH])).clip(aoi)

      b_red = LSImg.select('SR_B4_p'+str(percentReductionTH))
      b_nir = LSImg.select('SR_B5_p'+str(percentReductionTH))
      LSNdvi = b_nir.subtract(b_red).divide( b_nir.add(b_red).add(0.00001) ).rename('ndvi_'+str(i1))

      ndviList.append(LSNdvi)

      rgbImgList.append(LSImg.select(['SR_B4_p'+str(percentReductionTH),'SR_B3_p'+str(percentReductionTH),'SR_B2_p'+str(percentReductionTH)],['R','G','B']).multiply(0.0000275).add(-0.2))

    self.ndviStack = ee.Image(ndviList)
    self.rgbImgCollection = ee.ImageCollection(rgbImgList)

    self.aoi = aoi
    self.startYear = startYear
    self.endYear = endYear
  
  def getRGBImageByYear(self, year, verbose=True):

    idx1 = year - self.startYear
    imageTemp = ee.Image(self.rgbImgCollection.toList(self.rgbImgCollection.size().getInfo()).get(idx1))

    if verbose:
      print("## Selecting image for year " + str(year) + ".")
      print("## If image is cloudy, corresponding NDVI image can be removed before modelling using 'removeNDVIImageByYear' function.")

    return imageTemp

  def removeNDVIImageByYear(self, year, verbose=True):
    
    bandNames1 = self.ndviStack.bandNames()
    bandNames2 = bandNames1.filter(ee.Filter.neq('item', 'ndvi_'+str(year)))
    
    self.ndviStack = self.ndviStack.select(bandNames2)

    if verbose:
      print("## NDVI image for year " + str(year) + " was removed.")
    

  def generateGTData(self, numRandPixels, additionalGTPoints, verbose=True):
    
    randGTData = self.ndviStack.sample(region=self.aoi, numPixels=numRandPixels, scale=30)
    
    additionalGTData = self.ndviStack.sampleRegions(additionalGTPoints, scale=30)
    self.allGTData = randGTData.merge(additionalGTData)

    if verbose:
      print("## " + str(self.allGTData.size().getInfo()) + " GT points were created.")

  def clusterViaKMean(self, nClusters, verbose=True):
    
    self.nClusters = nClusters

    self.inputProperties = self.allGTData.first().propertyNames().getInfo()[:-1]

    self.clustererObj = ee.Clusterer.wekaKMeans(nClusters=nClusters).train(self.allGTData, inputProperties=self.inputProperties)
    self.ndviStackClustered = self.ndviStack.cluster(self.clustererObj).add(1) # make 0 class 1 to avoid conflict with nodata class
    
    if verbose:
      print("## Data is clustered with " + str(nClusters) + " Clusters via K-Mean.")
  
  def plotClusterCentersKMean(self):

    GTDataClusers = self.allGTData.cluster(self.clustererObj)

    for cluster_id in range(0,self.nClusters):

      GTDataClusers_1 = GTDataClusers.filter(ee.Filter.eq('cluster', cluster_id))

      trainPoints_class2 =  GTDataClusers_1.reduceColumns(ee.Reducer.median().repeat(len(self.inputProperties)), self.inputProperties)
      val1 = trainPoints_class2.getInfo()['median']

      plt.plot(self.inputProperties, val1, label=f'Cluster {cluster_id+1}') # make 0 class 1 to be consistant with above

    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("NDVI")
    plt.title("Cluster Centers")
    plt.xticks(rotation='vertical')
    plt.show()

  def clusterViaVQVAE(self, dataCSVPath, scalingPerc, nClusters, epochs, verbose=True):
    
    geemap.ee_to_csv(self.allGTData, filename=dataCSVPath)

    self.vqvae = VQ_VAE.VQ_VAE(dataCSVPath, scalingPerc, nClusters, ndviStack=self.ndviStack, epochs=epochs)

    if verbose:
      print("## Data is clustered with " + str(nClusters) + " Clusters via VQ-VAE.")

    self.ndviStackClustered = self.vqvae.getClusteredResult().add(1) # make 0 class 1 to avoid conflict with nodata class

    self.vqvae.plotLossCurve()

  def plotClusterCentersVQVAE(self):
    
    self.vqvae.plotClusterCentersVQVAE()

  def mergeRelevantClusters(self, clusterIDList, verbose=True):
    
    clusterMask = self.ndviStackClustered.eq(clusterIDList[0])

    for clusterID1 in clusterIDList[1:]:
      clusterMask = clusterMask.Or(self.ndviStackClustered.eq(clusterID1))

    self.ndviStackClusteredMerged = self.ndviStackClustered.updateMask(clusterMask).toShort()

    if verbose:
      print("## Clusters, except for " + str(clusterIDList) + " were masked out.")

  def getClusteredResult(self):
    return self.ndviStackClustered

  def getFinalResult(self):
    return self.ndviStackClusteredMerged

  def __monitor_task(self, task):
    
    print("Task submitted! Task ID:", task.id)
    spinner = ['—', '\\', '|', '/']  # Spinner characters

    while True:
      status = task.status()
      state = status['state']
      
      for char in spinner:
        print(f'\r{char} Exporting to GDrive...', end='', flush=True)
        time.sleep(1.0)

      if state == 'COMPLETED':
          print("\nExport completed successfully!")
          break
      elif state == 'FAILED':
          print("\nExport failed:", status['error_message'])
          break
      elif state == 'CANCELLED':
          print("\nExport cancelled by user.")
          break

  def exportOutput(self, outputFileName):

    task = ee.batch.Export.image.toDrive(
      image=self.ndviStackClusteredMerged,
      description=outputFileName,
      scale=30,
      region=self.aoi.geometry(),
      fileFormat='GeoTIFF',
      #skipEmptyTiles=True,
      maxPixels=1e13
    )
    task.start()

    self.__monitor_task(task)