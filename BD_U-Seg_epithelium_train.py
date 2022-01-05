import os
import time
import numpy as np

from skimage import io

from scipy.ndimage.morphology import binary_fill_holes, binary_erosion

import tensorflow as tf

#%% Parameters
# ROOT_PATH = 'C:/Datas/3-GitHub_BDehapiot/BD_U-Seg/data_40x/'
ROOT_PATH = 'E:/3-GitHub_BDehapiot/BD_U-Seg/data_40x/'
PATCH_SIZE = 128
PATCH_PAD = 8
TRAIN_MASK = 1

def train(root_path, patch_size, patch_pad, train_mask):
    

#%% Initialize

# Get train folder path (from ROOT_PATH)
train_path = (ROOT_PATH + "train/")

# Get pad_range (from PATCH_SIZE & PATCH_PAD)
pad_range = np.arange(
    (PATCH_SIZE-(PATCH_SIZE/PATCH_PAD))*-1,
    (PATCH_SIZE-(PATCH_SIZE/PATCH_PAD))+1,
    (PATCH_SIZE/PATCH_PAD)).astype('int') 

#%% Open training data 

# Create 'raw' and 'seg' lists
train_list_raw = []
train_list_seg = []
train_list = os.listdir(train_path) 
for name in train_list:
    if 'raw' in name: 
        train_list_raw.append(name)
    if 'seg' in name: 
        train_list_seg.append(name)

# Open training data        
start = time.time()
print("Open training data")
        
train_data = []     
for data_idx, (name_raw, name_seg) in enumerate(zip(train_list_raw, train_list_seg)):
    
    # Open raw and seg data
    train_raw = io.imread(train_path+name_raw)
    train_seg = io.imread(train_path+name_seg).astype('bool')

    # Check if raw data are from the same type
    data_type = (train_raw.dtype).name   
    if data_idx > 0 and train_data[data_idx-1][2] != data_type:
        raise Exception(
            'error: all raw data should be of the same type, ' + 
            train_data[data_idx-1][2] + ' and ' +  data_type + ' where found.')
    
    # Get variables
    ndim = (train_raw.ndim)
    if ndim == 2:
        nT = 1
        nY = train_raw.shape[0] # Get array dimension (y)
        nX = train_raw.shape[1] # Get array dimension (x)
        train_raw = train_raw.reshape((nT, nY, nX))
        train_seg = train_seg.reshape((nT, nY, nX))

    if ndim == 3:
        nT = train_raw.shape[0] # Get array dimension (t)
        nY = train_raw.shape[1] # Get array dimension (y)
        nX = train_raw.shape[2] # Get array dimension (x) 
    
    # Create mask data (if TRAIN_MASK == 1)
    if TRAIN_MASK == 1:
        train_mask = np.zeros([nT, nY, nX]).astype('bool')
        for time_idx in range(nT):       
            temp_seg = train_seg[time_idx,:,:]
            temp_mask = binary_fill_holes(temp_seg)
            temp_mask = binary_erosion(temp_mask)
            train_mask[time_idx,:,:] = temp_mask

    # Append train_data list
        train_data.append([train_raw, train_seg, data_type, nT, nY, nX, train_mask])
    else:
        train_data.append([train_raw, train_seg, data_type, nT, nY, nX, []])
    
end = time.time()
print(f"  {(end - start):5.3f} s")

#%% Patch training data

# Define training patches
start = time.time()
print("Define training patches")

pad_idx = 0
patch_coords = [] 
for data_idx in range(len(train_data)):
    
    # Get variables
    nT = train_data[data_idx][3]
    nY = train_data[data_idx][4]
    nX = train_data[data_idx][5]
    
    # Get y and x offsets
    y_offset = int((nY/PATCH_SIZE - nY//PATCH_SIZE) * PATCH_SIZE/2)
    x_offset = int((nX/PATCH_SIZE - nX//PATCH_SIZE) * PATCH_SIZE/2) 
    
    # Open mask data (if TRAIN_MASK == 1)
    if TRAIN_MASK == 1: 
        train_mask = train_data[data_idx][6]
    
    for time_idx in range(nT):
        
        pad_idx = -1
        
        for pad in pad_range: 
            
            pad_idx = pad_idx + 1
            
            # pad both y and x
            for y in range(y_offset, nY, PATCH_SIZE):
                for x in range(x_offset, nX, PATCH_SIZE):   
                    y_strt = y + pad; y_stop = y_strt + PATCH_SIZE 
                    x_strt = x + pad; x_stop = x_strt + PATCH_SIZE 
                        
                    # get full sized patch
                    if y_strt >= 0 and y_stop <= nY: 
                        if x_strt >= 0 and x_stop <= nX: 
                            if TRAIN_MASK == 0 or train_mask[
                                    time_idx, y_strt:y_stop, x_strt:x_stop
                                    ].all():
                                patch_coords.append([
                                    data_idx, time_idx, pad_idx, 
                                    y_strt, y_stop, x_strt, x_stop
                                    ])
            
            if pad != 0:
                
                pad_idx = pad_idx + 1
                
                # pad y only
                for y in range(y_offset, nY, PATCH_SIZE):
                    for x in range(x_offset, nX, PATCH_SIZE):   
                        y_strt = y + pad; y_stop = y_strt + PATCH_SIZE 
                        x_strt = x; x_stop = x_strt + PATCH_SIZE 
                            
                        # get full sized patch
                        if y_strt >= 0 and y_stop <= nY: 
                            if x_strt >= 0 and x_stop <= nX: 
                                if TRAIN_MASK == 0 or train_mask[
                                        time_idx, y_strt:y_stop, x_strt:x_stop
                                        ].all():
                                    patch_coords.append([
                                        data_idx, time_idx, pad_idx, 
                                        y_strt, y_stop, x_strt, x_stop
                                        ])
                
                pad_idx = pad_idx + 1
                
                # pad x only
                for y in range(y_offset, nY, PATCH_SIZE):
                    for x in range(x_offset, nX, PATCH_SIZE): 
                        y_strt = y; y_stop = y_strt + PATCH_SIZE 
                        x_strt = x + pad; x_stop = x_strt + PATCH_SIZE 
                            
                        # get full sized patch
                        if y_strt >= 0 and y_stop <= nY: 
                            if x_strt >= 0 and x_stop <= nX: 
                                if TRAIN_MASK == 0 or train_mask[
                                        time_idx, y_strt:y_stop, x_strt:x_stop
                                        ].all():
                                    patch_coords.append([
                                        data_idx, time_idx, pad_idx, 
                                        y_strt, y_stop, x_strt, x_stop
                                        ])

end = time.time()
print(f"  {(end - start):5.3f} s")  

# Extract training patches      
start = time.time()
print("Extract training patches")
                                                      
train_raw_patch = np.zeros([len(patch_coords),PATCH_SIZE,PATCH_SIZE])  
train_seg_patch = np.zeros([len(patch_coords),PATCH_SIZE,PATCH_SIZE]).astype('bool')        
for patch in range(len(patch_coords)):
    data_idx = patch_coords[patch][0] 
    time_idx = patch_coords[patch][1] 
    pad_idx = patch_coords[patch][2]                
    y_strt = patch_coords[patch][3]; y_stop = patch_coords[patch][4] 
    x_strt = patch_coords[patch][5]; x_stop = patch_coords[patch][6]       
    train_raw_patch[patch,:,:] = train_data[data_idx][0][time_idx,y_strt:y_stop,x_strt:x_stop]
    train_seg_patch[patch,:,:] = train_data[data_idx][1][time_idx,y_strt:y_stop,x_strt:x_stop]

end = time.time()
print(f"  {(end - start):5.3f} s") 

#%% Data augmentation

seed = 1

data_gen_args = dict(
    rotation_range = 20,
    width_shift_range = 0.2,   
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'reflect') 

raw_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
seg_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

raw_datagen.fit(np.expand_dims(train_raw_patch, axis=3), augment=True, seed=seed)
seg_datagen.fit(np.expand_dims(train_seg_patch, axis=3), augment=True, seed=seed)

test = raw_datagen.flow(np.expand_dims(train_raw_patch, axis=3), batch_size=32, seed=seed, save_to_dir = ROOT_PATH)

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rotation_range = 20,     
#         width_shift_range = 0.2,   
#         height_shift_range = 0.2,
#         shear_range = 0.2,
#         zoom_range = 0.2,
#         horizontal_flip = True,
#         vertical_flip = True,
#         fill_mode = 'reflect')  

#%% Network architecture (U-Net)

inputs = tf.keras.layers.Input((PATCH_SIZE, PATCH_SIZE, 1))

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansion path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

#%% Train the network (U-Net)

start = time.time()
print("Train the network")

# Normalize raw data
if train_data[0][2] == 'uint8':
    train_raw_patch = train_raw_patch/255
elif train_data[0][2] == 'uint16' or train_data[0][2] == 'float32':
    train_raw_patch = train_raw_patch/65535

# Compil the network
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]

# Train the network
results = model.fit(train_raw_patch, train_seg_patch, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)

# Save the model
from datetime import datetime
date_string = datetime.now().strftime('%b-%d-%Y_%Hh%M')
model_name = 'model_' + '{:04d}'.format(PATCH_SIZE) + '_' + '{:02d}'.format(PATCH_PAD) + '_' + date_string + '.h5'
model.save(ROOT_PATH + model_name)

end = time.time()
print(f"  {(end - start):5.3f} s") 

#%% Save data

# start = time.time()
# print("Save intermediate data")

# io.imsave(ROOT_PATH+'train_raw_patch.tif', train_raw_patch.astype("float32"), check_contrast=False)    
# io.imsave(ROOT_PATH+'train_seg_patch.tif', train_seg_patch.astype("uint8"), check_contrast=False) 

# end = time.time()
# print(f"  {(end - start):5.3f} s") 