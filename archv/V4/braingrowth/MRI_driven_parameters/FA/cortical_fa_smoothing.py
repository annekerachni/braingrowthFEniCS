import nibabel
import numpy
import skimage


age = '21'
seg = nibabel.load('/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t'+age+'.00_dhcp-19.nii.gz')
img = nibabel.load('/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t'+age+'.00.nii.gz')

seg_data = seg.get_fdata()
img_data = img.get_fdata()

cortical_fa = numpy.zeros(img_data.shape)
cortical_fa[seg_data == 3] = img_data[seg_data == 3]
cortical_fa[seg_data == 4] = img_data[seg_data == 4]

#nibabel.save(nibabel.Nifti1Image(cortical_fa, img.affine),'cor_fa_21.nii.gz')

norm_factor = numpy.max(cortical_fa)
new_cor_fa = skimage.filters.rank.maximum(cortical_fa/norm_factor, 
                                          footprint=skimage.morphology.ball(5), 
                                          mask = cortical_fa/norm_factor)

nibabel.save(nibabel.Nifti1Image(new_cor_fa/255.0*norm_factor, img.affine),'cortical_fa_'+age+'_max.nii.gz')