import nibabel
import numpy
import numpy as np

age = '27'
seg = nibabel.load('/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t'+age+'.00_dhcp-19.nii.gz')
fa = nibabel.load('/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t'+age+'.00.nii.gz')

seg_data = seg.get_fdata()
fa_data = fa.get_fdata()

# keep only cortical FA
cortical_fa = numpy.zeros(fa_data.shape)
cortical_fa[seg_data == 3] = fa_data[seg_data == 3]
cortical_fa[seg_data == 4] = fa_data[seg_data == 4]

# compute mean cortical FA (removing 0. value since it referes to background)
cortical_fa_vec = cortical_fa.flatten()
cortex_indices = np.where(cortical_fa_vec != 0.)[0] 

mean_value_FA = 0
for index in list(cortex_indices):
    mean_value_FA += cortical_fa_vec[index]
mean_value_FA /= len(cortex_indices)
print('mean FA = {}'.format(mean_value_FA) )
    
# save cortical FA
nibabel.save(cortical_fa,'cortical_fa_'+age+'.nii.gz')

# mean FA value =
# 21GW: 23.87585090697869
# 24GW: 24.507760241010224
# 27GW: 24.089925227540647
# 30GW: 
# 33GW: 
# 36GW: 