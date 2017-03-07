"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/4/2017

Various data visualization functions
"""
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_hu_histogram(scan_data):
    """
    Plots histogram of Hounsfield Units per pixel of a given scan.

    WARNING: This plotting function is blocking

    @param scan_data Scan array to visualize data for
    """
    plt.hist(scan_data.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # Show some slice in the middle
    plt.imshow(scan_data[80], cmap=plt.cm.gray)
    plt.show()


def plot_3d(image, threshold=-300, save_dir=None):
    """
    Generates and displays a 3D plot of the scan in an upright position.

    To view specific substances, refer to the following list of Hounsfield
    Unit equivalences for a threshold. For example, a threshold of 400 is
    good for viewing bones.
    https://en.wikipedia.org/wiki/Hounsfield_scale#The_HU_of_common_substances

    NOTE: To view segmented lungs, use a threshold of 0
    WARNING: This plotting function is blocking

    @param image The 3D image to plot
    @param threshold The threhold used to generate vertices and faces
    @param save_dir If not none, saves image to that path
    """
    # Position scan upright so patient head faces top of camera
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Indexing: 'verts[faces]' to generate collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir+'.jpg')
