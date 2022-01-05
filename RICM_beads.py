#%% Initialize
import numpy as np

from pystackreg import StackReg

from joblib import Parallel, delayed  

from skimage import io
from skimage.util import invert
from skimage.filters import sato
from skimage.measure import regionprops
from skimage.draw import circle_perimeter
from skimage.morphology import disk, black_tophat

from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage.morphology import distance_transform_edt

#%% varnames
ROOTPATH = 'E:/3-GitHub_BDehapiot/BD_RICMLib/'
RAWNAME = 'realcell_01_c01.tif'
EMPTY_t0 = 2242 # timepoint at which the bead is no longer visible
wAVG = 30 # window size for walking averaged imageSATO
SATO_sig = 3 # sigma size for sato filter 
pSEARCH = 20 # size in pixels for searching peak of interest

#%% Open Stack from RAWNAME

raw = io.imread(ROOTPATH+RAWNAME)
nT = raw.shape[0] # Get Stack dimension (t)
nY = raw.shape[1] # Get Stack dimension (x)
nX = raw.shape[2] # Get Stack dimension (y)

nTw = nT-wAVG # Get Stack dimension (t of walking averaged image)

#%% image processing

def image_process(im):
    '''Enter function general description + arguments'''
    strel = disk(5) 
    im_wavg = np.mean(im,0)
    im_binary = black_tophat(im_wavg, strel)
    im_binary = im_binary > np.max(im_binary)/2
    im_binary = im_binary.astype('float')
    return im_wavg, im_binary

output_list = Parallel(n_jobs=35)(
    delayed(image_process)(
        raw[i:i+wAVG,:,:]
        )
    for i in range(nTw)
    )
 
raw_wavg = np.stack([arrays[0] for arrays in output_list], axis=0)
raw_wavg_binary = np.stack([arrays[1] for arrays in output_list], axis=0)


#%% image registration (background)

def image_reg(im0, im1, im2reg):
    '''Enter function general description + arguments'''
    sr = StackReg(StackReg.TRANSLATION)
    sr.register(im0, im1)
    im_reg = sr.transform(im2reg)
    return im_reg

output_list = Parallel(n_jobs=35)(
    delayed(image_reg)(
        raw_wavg_binary[0,:,:],
        raw_wavg_binary[i,:,:],
        raw_wavg[i,:,:]) 
    for i in range(nTw)
    )
 
raw_wavg_reg = np.stack([arrays for arrays in output_list], axis=0)   

#%% subtract static background

static_bg = np.mean(raw_wavg_reg[EMPTY_t0:-1,:,:],0)
static_bg[static_bg == 0] = 'nan'
raw_wavg_reg[raw_wavg_reg == 0] = 'nan'
raw_wavg_reg_bgsub = raw_wavg_reg - static_bg
raw_wavg_reg_bgsub = np.nan_to_num(raw_wavg_reg_bgsub, nan=0.0) 

#%% apply sato filter

def sato_filter(im, sigmas):
    '''Enter function general description + arguments'''
    im_sato = sato(im,sigmas=sigmas,mode='reflect',black_ridges=False)   
    return im_sato

output_list = Parallel(n_jobs=35)(
    delayed(sato_filter)(
        raw_wavg_reg_bgsub[i,:,:],
        SATO_sig
        ) 
    for i in range(nTw)
    ) 

raw_wavg_reg_bgsub_sato = np.stack([arrays for arrays in output_list], axis=0)  

#%% image registration (bead)

# binarize sato filtered image
thresh_quant = np.quantile(raw_wavg_reg_bgsub_sato, 0.95)
raw_wavg_reg_bgsub_sato = raw_wavg_reg_bgsub_sato > thresh_quant
raw_wavg_reg_bgsub_sato = raw_wavg_reg_bgsub_sato.astype('float') 
           
output_list = Parallel(n_jobs=35)(
    delayed(image_reg)(
        raw_wavg_reg_bgsub_sato[0,:,:],
        raw_wavg_reg_bgsub_sato[i,:,:],
        raw_wavg_reg_bgsub[i,:,:]) 
    for i in range(nTw)
    )
 
raw_wavg_reg_bgsub_reg = np.stack([arrays for arrays in output_list], axis=0)     

#%% circular averaging

# get circle centroid (first frame as reference)
props = regionprops(raw_wavg_reg_bgsub_sato[0,:,:].astype('int'))
ctrd_x = np.round(props[0].centroid[1]).astype('int')
ctrd_y = np.round(props[0].centroid[0]).astype('int')

# get Euclidian distance map (edm)
centroid = np.zeros([nY, nX])
centroid[ctrd_y, ctrd_x] = 1
centroid_edm = distance_transform_edt(invert(centroid))

def circular_avg(im):
    unique = np.unique(centroid_edm)
    unique_meanval = np.zeros([len(unique)])    
    im_circavg = np.zeros([nY, nX])
    for i in range(len(unique)):
        tempidx = np.where(centroid_edm == unique[i])
        unique_meanval[i] = np.mean(im[tempidx])
        im_circavg[tempidx] = unique_meanval[i]
    return im_circavg

output_list = Parallel(n_jobs=35)(
    delayed(circular_avg)(
        raw_wavg_reg_bgsub_reg[i,:,:],
        ) 
    for i in range(nTw)
    )
 
raw_wavg_reg_bgsub_reg_circavg = np.stack([arrays for arrays in output_list], axis=0) 

#%% apply sato filter

output_list = Parallel(n_jobs=35)(
    delayed(sato_filter)(
        raw_wavg_reg_bgsub_reg_circavg[i,:,:],
        SATO_sig
        ) 
    for i in range(nTw)
    ) 

raw_wavg_reg_bgsub_reg_circavg_sato = np.stack([arrays for arrays in output_list], axis=0)  

#%% 

temp_profile = raw_wavg_reg_bgsub_reg_circavg_sato[0,ctrd_y, ctrd_x:ctrd_x+pSEARCH]
profile = np.zeros([len(temp_profile), nTw])
profile_interp = np.zeros([len(temp_profile)*100, nTw])
xinterp = np.linspace(0, pSEARCH-1, num=len(temp_profile)*100, endpoint=True) 
peak_of_interest = np.zeros([nTw])*np.nan
for i in range(nTw):
    profile[:,i] = raw_wavg_reg_bgsub_reg_circavg_sato[i,ctrd_y, ctrd_x:ctrd_x+20]  
    x = np.linspace(0, pSEARCH-1, num=len(temp_profile), endpoint=True)      
    f = interp1d(x, profile[:,i], kind='cubic')
    profile_interp[:,i] = f(xinterp)
    peaks, properties = find_peaks(profile_interp[:,i], prominence=0.1)
    if peaks.size:
        peak_of_interest[i] = xinterp[peaks[0]]
    else:
        peak_of_interest[i] = np.nan        

tracking_display = np.zeros([nTw,nY,nX])  
for i in range(nTw):
    rr, cc = circle_perimeter(ctrd_y,ctrd_x, (peak_of_interest[i]*1.5).astype('int'), shape=tracking_display[i,:,:].shape)
    tracking_display[i,:,:][rr, cc] = 1 
    

temp_nan_poi = np.zeros([wAVG//2])*np.nan 
peak_of_interest = np.concatenate((
    temp_nan_poi, 
    peak_of_interest, 
    temp_nan_poi),
    axis=0
    ) 

temp_nan_im = np.zeros([wAVG//2,nY,nX])        
raw_wavg_reg_bgsub_reg = np.concatenate((
    temp_nan_im, 
    raw_wavg_reg_bgsub_reg, 
    temp_nan_im), 
    axis=0
    ) 
tracking_display = np.concatenate((
    temp_nan_im, 
    tracking_display, 
    temp_nan_im), 
    axis=0
    ) 

#%% Export results

np.savetxt(ROOTPATH+RAWNAME[0:-4]+"_inner_circle_diameter_pixel.csv", peak_of_interest, fmt='%.4f', delimiter=",")

#%% Save images

io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg.tif', raw_wavg.astype('uint8'), check_contrast=True)
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_binary.tif', raw_wavg_binary.astype('uint8')*255, check_contrast=True)
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg.tif', raw_wavg_reg.astype('uint8'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub.tif', raw_wavg_reg_bgsub.astype('float32'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_sato.tif', raw_wavg_reg_bgsub_sato.astype('float32'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_reg.tif', raw_wavg_reg_bgsub_reg.astype('float32'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_reg_circavg.tif', raw_wavg_reg_bgsub_reg_circavg.astype('float32'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_reg_circavg_sato.tif', raw_wavg_reg_bgsub_reg_circavg_sato.astype('float32'), check_contrast=True) 
io.imsave(ROOTPATH+RAWNAME[0:-4]+'_tracking_display.tif', tracking_display.astype('uint8')*255, check_contrast=True)