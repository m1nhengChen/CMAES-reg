import nibabel as nib
import numpy as np
import pydicom
import os
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing, binary_opening
import vedo
import pyvista as pv
# Step 1: Load 3D CT data
def load_nifti(file_path):
    """Load NIfTI format CT image"""
    nii_img = nib.load(file_path)
    return nii_img.get_fdata()

def load_dicom(dicom_folder):
    """Load DICOM files and sort slices based on Z-axis"""
    slices = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in os.listdir(dicom_folder)]
    slices.sort(key=lambda x: x.ImagePositionPatient[2])  # Sort by Z-axis
    return np.stack([s.pixel_array for s in slices], axis=-1)

# Load CT scan (choose the appropriate method)
ct_scan = load_nifti("../512/all/CTJ130564_512.nii.gz")
# ct_scan = load_dicom("dicom_data/")
# ct_scan = np.random.randint(0, 1000, (512, 512, 100))  # Dummy data for testing

# Step 2: Preprocessing (Denoising and smoothing)
ct_scan_smoothed = gaussian_filter(ct_scan, sigma=1)

# Step 3: Segment vertebral body using Hounsfield Unit (HU) threshold
bone_threshold = (200, 1000)  # HU range for bones
binary_mask = (ct_scan_smoothed > bone_threshold[0]) & (ct_scan_smoothed < bone_threshold[1])

# Apply morphological operations to remove small artifacts
cleaned_mask = binary_opening(binary_mask, structure=np.ones((3,3,3)))  # Remove noise
binary_mask = binary_closing(cleaned_mask, structure=np.ones((3,3,3)))  # Fill small gaps
# Step 4: Extract individual vertebral body using connected components analysis
labeled_mask, num_labels = ndi.label(binary_mask)
sizes = np.bincount(labeled_mask.ravel())
largest_label = sizes[1:].argmax() + 1  # Ignore background label (0)
vertebral_body_mask = (labeled_mask == largest_label)
smoothed_mask = gaussian(vertebral_body_mask.astype(float), sigma=1)
# Step 5: Extract 3D contour points using Marching Cubes algorithm
verts, faces, _, _ = measure.marching_cubes(smoothed_mask, level=0)
contour_points_3d = verts  # Extract contour points

# # Step 6: Visualize the 3D contour points
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")

# # Scatter plot of contour points
# ax.scatter(contour_points_3d[:, 0], contour_points_3d[:, 1], contour_points_3d[:, 2], s=1, c="b")

# plt.show()

# # Step 7: Visualize 3D surface mesh
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")

# # Generate 3D mesh surface
# mesh = Poly3DCollection(contour_points_3d[faces], alpha=0.3)
# ax.add_collection3d(mesh)

# ax.set_xlim(0, vertebral_body_mask.shape[0])
# ax.set_ylim(0, vertebral_body_mask.shape[1])
# ax.set_zlim(0, vertebral_body_mask.shape[2])

# plt.show()
plotter = vedo.Plotter()
cloud = vedo.Points(contour_points_3d, r=2, c="grey")  # Adjust point size (r) and color (c)

# Show interactive 3D rendering
plotter.show(cloud, "3D Vertebral Contour", axes=True)

