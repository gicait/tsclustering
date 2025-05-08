// UI Setup
var panelL = ui.Panel({layout: ui.Panel.Layout.flow('vertical'), style: {width: '300px', position: 'top-left', border: '1px solid gray'}});
ui.root.insert(0, panelL);
panelL.add(ui.Label('Landsat 8 - Time Series Clustering', {fontWeight: 'bold', fontSize: '16px'}));

panelL.add(ui.Label({value:'Instructions:', style:{fontStyle: 'italic', fontSize: '12px'}}));
panelL.add(ui.Label({value:'1. Draw a polygon to define the study area.', style:{fontStyle: 'italic', fontSize: '12px'}}));
panelL.add(ui.Label({value:'2. Adjust the parameters below, and click the "Run" button to start the clustering process.', style:{fontStyle: 'italic', fontSize: '12px'}}));

panelL.add(ui.Label('Input Parameters', {fontWeight: 'bold'}));

// Date input
var startYearUI = ui.Slider({min: 2013, max: 2025, value: 2013, step: 1, style: {stretch: 'horizontal'}});
var endYearUI = ui.Slider({min: 2013, max: 2025, value: 2025, step: 1, style: {stretch: 'horizontal'}});

panelL.add(ui.Label('Start Year of the Analysis (2013-2025):'));
panelL.add(startYearUI);
panelL.add(ui.Label('End Year of the Analysis (2013-2025):'));
panelL.add(endYearUI);

// Cloud cover input
var cloudTHUI = ui.Slider({min: 0, max: 100, value: 20, step: 1, style: {stretch: 'horizontal'}});
panelL.add(ui.Label('Maximum Cloud Cover (0-100%) allowed in the Analysis:'));
panelL.add(cloudTHUI);

// Percentile slider
var percentReductionTHUI = ui.Slider({min: 0, max: 100, value: 50, step: 1, style: {stretch: 'horizontal'}});
panelL.add(ui.Label('Select a Percentile (0–100) value for yearly percentile reduction (creating a single image for each year):'));
panelL.add(percentReductionTHUI);

// Number of random samples
var numRandPixelsUI = ui.Slider({min: 1000, max: 10000, value: 1000, step: 500, style: {stretch: 'horizontal'}});
panelL.add(ui.Label('Number of random sample to be taken within AOI (1,000-10,000):'));
panelL.add(numRandPixelsUI);

// Max number of clusters
var nClustersUI = ui.Slider({min: 2, max: 15, value: 5, step: 1, style: {stretch: 'horizontal'}});
panelL.add(ui.Label('Maximum number of expected clusters (2-15):'));
panelL.add(nClustersUI);

// Select algorithm
var cAlgoUI = ui.Select({items: ['KMeans','CascadeKMeans','XMeans','LVQ'],  value:'KMeans', placeholder: 'Select a algorithm',});
panelL.add(ui.Label('Choose a clustering algorithm:'));
panelL.add(cAlgoUI);

// Button
var processButton = ui.Button({
  label: 'Run',
  style: {stretch: 'horizontal'},
  onClick: runAnalysis
});
panelL.add(processButton);

panelL.add(ui.Label({value: 'Output Notes:', style: {fontWeight: 'bold'}}));

// Draw geometry
var aoiTool = Map.drawingTools();
aoiTool.setShown(true);
aoiTool.setDrawModes(['rectangle', 'polygon']);
aoiTool.setLinked(true);

/*
// Clear existing layers and geometries
function clearGeometry() {
  aoiTool.layers().reset();
  aoiTool.setDrawMode('polygon');
}
clearGeometry();
*/

Map.setCenter(104.9910, 12.5657, 7);

// Run analysis
function runAnalysis() {
  
  var layers = aoiTool.layers();
  if (layers.length() === 0) {
    panelL.add(ui.Label({value:'ERROR: Please draw a polygon to define the study area!', style:{fontStyle: 'italic'}}));
    return;
  }
  
  var aoi = aoiTool.layers().get(0).getEeObject();
  //var aoi = ee.Geometry.Rectangle([104.9910-0.1, 12.5657-0.1, 104.9910+0.1, 12.5657+0.1]);
  
  var startYear = startYearUI.getValue();
  var endYear = endYearUI.getValue();
  var cloudTH = cloudTHUI.getValue();
  var percentReductionTH = percentReductionTHUI.getValue();
  var numRandPixels = numRandPixelsUI.getValue();
  var nClusters = nClustersUI.getValue();
  var cAlgo = cAlgoUI.getValue();
  
  if (Number(startYear) > Number(endYear)) {
    panelL.add(ui.Label({value:'ERROR: End year of the analysis must be larger than start year of the analysis!', style:{fontStyle: 'italic'}}));
    return;
  }
  
  var ndviList = [];
  var rgbImgList = [];
  
  var cloudMask = function(img) {
    var qaMask = img.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
    var saturationMask = img.select('QA_RADSAT').eq(0);
  
    return img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']).updateMask(qaMask).updateMask(saturationMask);
  };
  
  var noImageYears = '';
  
  for (var i1 = startYear; i1 <= endYear; i1++) {
    var LSCol = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterDate(i1 + '-01-01', i1 + '-12-31')
      .filterBounds(aoi)
      .filterMetadata('CLOUD_COVER', 'less_than', cloudTH);
    
    if (Number(LSCol.size().getInfo()) === 0) {
      noImageYears = noImageYears + i1.toString() + ', ';
    }
    
    LSCol = LSCol.map(cloudMask);
    
    var LSImg = LSCol.select(['SR_B2','SR_B3','SR_B4','SR_B5'])
    .reduce(ee.Reducer.percentile([percentReductionTH]))
    .clip(aoi);
    
    var b_red = LSImg.select('SR_B4_p' + percentReductionTH);
    var b_nir = LSImg.select('SR_B5_p' + percentReductionTH);

    // Calculate NDVI
    var LSNdvi = b_nir.subtract(b_red).divide(b_nir.add(b_red).add(0.00001)).rename('ndvi_' + i1);
  
    // Append NDVI image to list
    ndviList.push(LSNdvi);
    
    var rgbImg = LSImg.select(
      ['SR_B4_p' + percentReductionTH, 'SR_B3_p' + percentReductionTH, 'SR_B2_p' + percentReductionTH],
      ['R', 'G', 'B']
    ).multiply(0.0000275).add(-0.2);
    
    rgbImgList.push(rgbImg);
  }
  
  if (noImageYears !== '') {
    panelL.add(ui.Label({value:'ERROR: No images found in these years: ' + noImageYears + '. Adjust above parameters and re-run.', style:{fontStyle: 'italic'}}));
    return;
  }
  
  var ndviStack = ee.Image(ndviList);
  var rgbImgCollection = ee.ImageCollection(rgbImgList);
  
  var randGTData = ndviStack.sample({
    region: aoi,
    numPixels: Number(numRandPixels),
    scale: 30
  });
  
  // Get input property names for clustering
  var inputProperties = randGTData.first().propertyNames();
  inputProperties = inputProperties.slice(0, inputProperties.length().subtract(1));
  
  // Select clustering algorithm
  var clustererAlg;
  if (cAlgo === 'KMeans') {
    clustererAlg = ee.Clusterer.wekaKMeans(Number(nClusters));
  } else if (cAlgo === 'CascadeKMeans') {
    clustererAlg = ee.Clusterer.wekaCascadeKMeans(2, Number(nClusters));
  } else if (cAlgo === 'XMeans') {
    clustererAlg = ee.Clusterer.wekaXMeans(2, Number(nClusters));
  } else if (cAlgo === 'LVQ') {
    clustererAlg = ee.Clusterer.wekaLVQ(Number(nClusters));
  } 
  
  // Train the clusterer
  var clustererObj = clustererAlg.train({
    features: randGTData,
    inputProperties: inputProperties
  });
  
  // Apply clustering to NDVI stack
  var ndviStackClustered = ndviStack.cluster(clustererObj).add(1); // shift class labels by 1
  
  var palette = [
    '05450a',
    '78d203',
    '009900',
    'c6b044',
    'dcd159',
    'dade48',
    'fbff13',
    'b6ff05',
    '27ff87',
    'c24f44',
    'a5a5a5',
    'ff6d4c',
    '69fff8',
    'f9ffa4',
    '1c0dff',
  ]; // Random color palette
  
  panelL.add(ui.Label({value: 'The ' + cAlgo + ' clustering process has completed !!!', style: {fontWeight: 'bold'}}));
  
  Map.centerObject(aoi, 7);
  Map.layers().set(0, ui.Map.Layer(ndviStackClustered, {min: 1, max: Number(nClusters)+1, palette: palette}, 'Clustered Result'));
  
}
