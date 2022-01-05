import napari
import time
import numpy as np

from skimage import io

import tensorflow as tf

#%% Parameters
# ROOT_PATH = 'C:/Datas/3-GitHub_BDehapiot/BD_U-Seg/data_40x/'
ROOT_PATH = 'E:/3-GitHub_BDehapiot/BD_U-Seg/data_40x/'
MODEL_NAME = 'model_0128_08_Sep-03-2021_14h50.h5'
RAW_NAME = '18-07-11_GBE_67xYW(F2)_b1_StackCrop_raw.tif'
PATCH_PAD = 8

#%% Initialize

# Load model.h5
model = tf.keras.models.load_model(ROOT_PATH + MODEL_NAME)
patch_size = model.layers[0].input_shape[0][1]

# Get pad_range (from patch_size & PATCH_PAD)
pad_range = np.arange(
    (patch_size-(patch_size/PATCH_PAD))*-1,
    (patch_size-(patch_size/PATCH_PAD))+1,
    (patch_size/PATCH_PAD)).astype('int')

#%% Open raw data 

start = time.time()
print("Open raw data")
       
raw_data = []  
    
# Open raw data
raw = io.imread(ROOT_PATH + RAW_NAME)  

# Get data type
data_type = (raw.dtype).name   

# Get variables
ndim = (raw.ndim)
if ndim == 2:
    nT = 1
    nY = raw.shape[0] # Get array dimension (y)
    nX = raw.shape[1] # Get array dimension (x)
    raw = raw.reshape((nT, nY, nX))

if ndim == 3:
    nT = raw.shape[0] # Get array dimension (t)
    nY = raw.shape[1] # Get array dimension (y)
    nX = raw.shape[2] # Get array dimension (x)
    
# Append raw_data list
raw_data.append([raw, [], data_type, nT, nY, nX])
        
end = time.time()
print(f"  {(end - start):5.3f} s")

#%% Patch raw data

# Define raw patches
start = time.time()
print("Define raw patches")

pad_idx = 0
patch_coords = [] 
for data_idx in range(len(raw_data)):
    
    # Get variables
    nT = raw_data[data_idx][3]
    nY = raw_data[data_idx][4]
    nX = raw_data[data_idx][5]
    
    # Get y and x offsets
    y_offset = int((nY/patch_size - nY//patch_size) * patch_size/2)
    x_offset = int((nX/patch_size - nX//patch_size) * patch_size/2) 
        
    for time_idx in range(nT):
        
        pad_idx = -1
        
        for pad in pad_range: 
            
            pad_idx = pad_idx + 1
            
            # pad both y and x
            for y in range(y_offset, nY, patch_size):
                for x in range(x_offset, nX, patch_size):   
                    y_strt = y + pad; y_stop = y_strt + patch_size 
                    x_strt = x + pad; x_stop = x_strt + patch_size 
                        
                    # get full sized patch
                    if y_strt >= 0 and y_stop <= nY: 
                        if x_strt >= 0 and x_stop <= nX: 
                            patch_coords.append([
                                data_idx, time_idx, pad_idx, 
                                y_strt, y_stop, x_strt, x_stop
                                ])
            
            if pad != 0:
                
                pad_idx = pad_idx + 1
                
                # pad y only
                for y in range(y_offset, nY, patch_size):
                    for x in range(x_offset, nX, patch_size):   
                        y_strt = y + pad; y_stop = y_strt + patch_size 
                        x_strt = x; x_stop = x_strt + patch_size 
                            
                        # get full sized patch
                        if y_strt >= 0 and y_stop <= nY: 
                            if x_strt >= 0 and x_stop <= nX: 
                                patch_coords.append([
                                    data_idx, time_idx, pad_idx, 
                                    y_strt, y_stop, x_strt, x_stop
                                    ])
                
                pad_idx = pad_idx + 1
                
                # pad x only
                for y in range(y_offset, nY, patch_size):
                    for x in range(x_offset, nX, patch_size): 
                        y_strt = y; y_stop = y_strt + patch_size 
                        x_strt = x + pad; x_stop = x_strt + patch_size 
                            
                        # get full sized patch
                        if y_strt >= 0 and y_stop <= nY: 
                            if x_strt >= 0 and x_stop <= nX: 
                                patch_coords.append([
                                    data_idx, time_idx, pad_idx, 
                                    y_strt, y_stop, x_strt, x_stop
                                    ])

end = time.time()
print(f"  {(end - start):5.3f} s")  

# Extract training patches     
start = time.time()
print("Extract predictation patches")
                                                      
raw_patch = np.zeros([len(patch_coords),patch_size,patch_size]).astype('uint16')        
for patch in range(len(patch_coords)):
    data_idx = patch_coords[patch][0] 
    time_idx = patch_coords[patch][1] 
    pad_idx = patch_coords[patch][2]                
    y_strt = patch_coords[patch][3]; y_stop = patch_coords[patch][4] 
    x_strt = patch_coords[patch][5]; x_stop = patch_coords[patch][6]       
    raw_patch[patch,:,:] = raw_data[data_idx][0][time_idx,y_strt:y_stop,x_strt:x_stop]

end = time.time()
print(f"  {(end - start):5.3f} s") 

#%% Predict raw data

# Normalize raw data
if raw_data[0][2] == 'uint8':
    raw_patch = raw_patch/255
elif raw_data[0][2] == 'uint16' or raw_data[0][2] == 'float32':
    raw_patch = raw_patch/65535

# Predict raw patches
predict_patch = model.predict(raw_patch, verbose=1) 
predict_patch = np.squeeze(predict_patch) # get rid of len = 1 dimensions

# Assemble prediction image (from patches)
start = time.time()
print("Assemble prediction image")

border_offset = patch_size//(PATCH_PAD*2) 
predict = np.empty([len(pad_range) + 2*(len(pad_range)-1), nT, nY, nX])
predict[:] = np.NaN
for patch in range(len(patch_coords)):
    time_idx = patch_coords[patch][1] 
    pad_idx = patch_coords[patch][2]                
    y_strt = patch_coords[patch][3] + border_offset
    y_stop = patch_coords[patch][4] - border_offset
    x_strt = patch_coords[patch][5] + border_offset
    x_stop = patch_coords[patch][6] - border_offset  
    temp_predict = predict_patch[patch, 
        border_offset:patch_size-border_offset, 
        border_offset:patch_size-border_offset]                                    
    predict[pad_idx, time_idx, y_strt:y_stop, x_strt:x_stop] = temp_predict
    
predict = np.nanmean(predict, axis=0)

end = time.time()
print(f"  {(end - start):5.3f} s") 

#%% Display patching coverage

start = time.time()
print("Display patching coverage")

border_offset = patch_size//(PATCH_PAD*2)          
patch_coverage = np.zeros([len(pad_range) + 2*(len(pad_range)-1), nY, nX])
for patch in range(len(patch_coords)):
    pad_idx = patch_coords[patch][2]                
    y_strt = patch_coords[patch][3] + border_offset
    y_stop = patch_coords[patch][4] - border_offset 
    x_strt = patch_coords[patch][5] + border_offset
    x_stop = patch_coords[patch][6] - border_offset      
    patch_coverage[pad_idx,y_strt:y_stop,x_strt:x_stop] = 1
    
patch_coverage = np.sum(patch_coverage, axis=0) 

end = time.time()
print(f"  {(end - start):5.3f} s")  

#%% Save data

start = time.time()
print("Save prediction data")

io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_predict.tif', predict.astype("float32"), check_contrast=False) 

end = time.time()
print(f"  {(end - start):5.3f} s") 

# start = time.time()
# print("Save intermediate data")

# io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_predict_patch.tif', predict_patch.astype("float32"), check_contrast=False) 
# io.imsave(ROOT_PATH+RAW_NAME[0:-4]+'_patch_coverage.tif', patch_coverage.astype('uint8'), check_contrast=False) 

# end = time.time()
# print(f"  {(end - start):5.3f} s") 

#%% Napari implementation (current)

viewer = napari.view_image(raw[34,:,:], name='raw')
viewer.add_image(wat[34,:,:].astype('uint8')*255, name='segmentation', colormap='yellow', blending='additive')
viewer.add_labels(np.zeros([nY,nX]).astype('uint8'), name='correction')

# stop

# correction = viewer.layers['correction'].data