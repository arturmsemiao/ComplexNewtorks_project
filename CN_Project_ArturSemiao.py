# importing required libraries
import matplotlib.pyplot as plt
from PIL import Image 
import matplotlib.image as img
import numpy as np
import math
import scipy
import time
import krypy 
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
from scipy.ndimage import gaussian_filter, standard_deviation
from sklearn.feature_extraction import image
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

def add_AGWN(img, mean_noise, std_noise, do_plot = True):
    """
    Adds additive gaussian white noise to a grey-scale image 'img', and computes RMSE of noise corrupted image

    Aditional arguments:
    'mean_noise': mean of noise, float
    'std_noise' : standard deviation of noise, float
    'do_plot': If true, makes plot of ground truth and noise corrupted images, side by side. Boolean

    Returns: noise corrupted image and respective rmse value, matrix and float.
    """

    img = img/255  #Scale pixels to range between 0 and 1
    noise = np.random.normal(loc = mean_noise, scale = std_noise, size = img.shape)
    noisy_image = img + noise
    noisy_image[noisy_image > 1] = 1
    noisy_image[noisy_image < 0] = 0
    noisy_image*= 255

    rmse = mean_squared_error(img*255, noisy_image, squared = False)
    print("RMS error noisy image: ",np.round(rmse,2))
    
    if do_plot:

        f, axarr = plt.subplots(1,2)
        axarr[0].set_title('a')
        axarr[0].imshow(img*255, cmap = 'gray', vmin = 0, vmax = 255)
        axarr[1].set_title('b')
        axarr[1].imshow(noisy_image, cmap = 'gray', vmin = 0, vmax = 255)    
        plt.show()

    return noisy_image, rmse

def Compute_Compressed_Laplacian_Matrix(img, std_GWED, kappa, patch_size):
    """

    Computes the compressed Lalplacian Matrix for a given image 'img'.

    Additional arguments:
    'std_GWED': standard deviation to compute Gaussian weighted Euclidean distances, float number
    'kappa' : kappa fator for the weights function, float number
    'patch_size': size of the patch to compute windoes similarity, int number

    Returns: Compressed Laplacian Matrix, matrix

    """
    print('std GWED: ',std_GWED,' kappa: ', kappa,' patch_size: ', patch_size)
    n_pixels = img.shape[0]*img.shape[1]
    
    #Create the patche
    r = 1 #distance threshold between two neighbouring pixels, controls local connectivity of the graph
    neighbour_size = 2*r + 1 #full lenght of the neighborhood envolving a pixel
    neg_padded_image = np.pad(np.arange(n_pixels).reshape(img.shape), r, mode = 'constant', constant_values = -1) #pad image with -1 to take care of borders
    pixels_neighbourhoods = image.extract_patches_2d(neg_padded_image, (neighbour_size , neighbour_size))
    
    Weights_Matrix = np.zeros((n_pixels, neighbour_size**2)) #Create the (compressed) weights matrix
    
    patch_radius = int((patch_size-1)/2) #radius of patch surrounding the pixel
    zero_padded_image = np.pad(img, patch_radius , mode= 'constant') #pad image with zeros to take care of borders 
    image_patches = image.extract_patches_2d(zero_padded_image, (patch_size, patch_size))

    #Create Gauss kernel with size equal to the patch size and std deviation equal to  'std_GWED'
    gauss_kernel = gkern(l = patch_size, sig = std_GWED)
    #Create now diagonal matrix which will be used in Gaussian weighted euclidean distance
    diag_gauss_kernel = np.zeros((patch_size**2, patch_size**2))
    np.fill_diagonal(diag_gauss_kernel, gauss_kernel.flatten())

    
    for pixel in range(n_pixels):

        pixel_patch = image_patches[pixel] #Get the patch of the pixel i
        pixel_neighbourhood = pixels_neighbourhoods[pixel].flatten() #Get the neighbourhood of the pixel i
        
        for i, adjacent_pixel in enumerate(pixel_neighbourhood):
            
            #IF THE ADJACENT PIXEL IS EQUAL TO PIXEL OR IS OUTSIDE THE BORDER, i.e. -1, OF THE IMAGE DO NOT COMPUTE WEIGHTS!
            if np.logical_or(adjacent_pixel < 0, adjacent_pixel == pixel): continue 
            adjacent_pixel_patch = image_patches[int(adjacent_pixel)] #Get the patch of the adjacent pixel j

            #COMPUTE SIMILARITY BETWEEN PIXEL PATCH AND ADJ PIXEL PATCH!! COMPUTE GAUSSIAN WEIGHTED EUCLIDEAN DISTANCE (GWED) 
            patches_intensity_difference = pixel_patch.flatten() - adjacent_pixel_patch.flatten() 
            GWED = np.dot( patches_intensity_difference, np.matmul(diag_gauss_kernel, patches_intensity_difference) )                    
            Weights_Matrix[pixel,i] = np.sqrt(GWED) #Add to the 'Weights Matrix'

    #Set the minimum distance to be 0 and the maximum distance to be 1, and rescale the intermediate values accordingly.
    #Only consider values !=0 since 0 values are the weights between pixel in border and pad pixels.
    min_weight, max_weight = np.min(Weights_Matrix[Weights_Matrix != 0.]), np.max(Weights_Matrix)    
    Weights_Matrix[Weights_Matrix != 0.] = (Weights_Matrix[Weights_Matrix != 0.] - min_weight)/(max_weight-min_weight)     
    #Compute real weights from the weight function for a given value of kappa
    Weights_Matrix[Weights_Matrix != 0.] = np.exp(- np.square( Weights_Matrix[Weights_Matrix != 0.]/kappa ) )
    
    #Build the compressed Laplacian Matrix
    Compressed_Laplacian_Matrix = - Weights_Matrix #Non diagonal entries of Laplacian are the inverse weights 
    Compressed_Laplacian_Matrix[:, neighbour_size**2//2] = np.sum(Weights_Matrix, axis = 1) #Diagonal entry, the middle one, is the degree of each pixel, given my sum of weights
          
    return Compressed_Laplacian_Matrix

def gkern(l, sig):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def Laplacian_Times_Vector(Laplacian, vector, image_shape):
    """
    This function serves to multiply the compressed (n_pixel, (2r+1)**2) Laplacian matrix by a (n_pixel,1) vector

    Arguments:
    'Laplacian': Compressed Laplacian to be multiplied, (n_pixel, (2r+1)**2) matrix
    'vector' : vector with lenght equal to number of pixels, (vectorized image) 
    'image_shape': shape of original image, to build neighbourhood patches, tuple with |rows| and |columns| of an image 

    Returns: Resulting vector of compressed multiplication
    """
    n_pixels = len(vector)
    padded_vector = np.pad(vector, [(0,1)], mode = 'constant') #Build a padded vector with an extra 0 in the end of original vector.
    
    neighbourhood_size = int(np.sqrt(Laplacian.shape[1]))
    neg_padded_image = np.pad(np.arange(n_pixels).reshape(image_shape),int((neighbourhood_size-1)/2), mode = 'constant', constant_values = -1)
    pixels_neighbourhoods = image.extract_patches_2d(neg_padded_image, (neighbourhood_size , neighbourhood_size))

    resulting_vector = np.zeros(n_pixels)

    for pixel in range(n_pixels):
        
        pixel_neighbourhood = pixels_neighbourhoods[pixel].flatten().astype(int)    
        resulting_vector[pixel] = np.dot( Laplacian[pixel], padded_vector[pixel_neighbourhood])
        
    return resulting_vector
  

def arnoldi_method( Compressed_Laplacian, noisy_img, m, diff_time):
    """
    This function does executes de arnoldi method to apply the Krylov subspace technique to smooth the noisy image

    Arguments:
    'Compressed Laplacian': Compressed Laplacian to be multiplied, matrix
    'noisy_img' : noisy image to be smoothed, matrix
    'm': iteration factor, import to Krylov subspace base, int number
    'diff_time' : max diffusion time, int number

    Returns: Array of smoothed images for each second of diffusion time
    """
    
    itensities_vector = noisy_img.flatten()
    eps = 1e-12
    t = 0

    n_pixels = noisy_img.shape[0]*noisy_img.shape[1]
    
    Krylov_Base = np.zeros( (n_pixels, m+1) )  #(n_pixel,m+1) matrix -> base of Krylov subspace
    Hessenber_Matrix = np.zeros( (m+2 , m+2) ) #Create an (m+2,m+2) matrix

    smoothed_images = []
    
    while t < diff_time:
        
        #First element of Krylov supbspace base is the original normalized vectorized image.
        beta = np.linalg.norm(itensities_vector) #Norm of the initial condition image
        Krylov_Base[:,0] = itensities_vector / beta 

        for j in range(m): #j goes from 0 to m-1
            
            p = Laplacian_Times_Vector(Compressed_Laplacian, Krylov_Base[:,j], noisy_img.shape) #(n_pixel,n_pixel)*(n_pixel,1) object = (n_pixel,1) object

            for i in range(j+1): #goes from 0 to j
        
                Hessenber_Matrix[i,j] = np.dot(Krylov_Base[:, i], p)  #Multiplication of (1,n_pixel)*(n_pixel,1) = number! 
                p -= Hessenber_Matrix[i,j] * Krylov_Base[:,i] #(n_pixel,1) - (number)*(n_pixel,1) = (n_pixel,1) object
            
            Hessenber_Matrix[j+1,j] = np.linalg.norm(p)
            if np.abs(Hessenber_Matrix[j+1,j]) < eps: continue
            Krylov_Base[:, j+1] = p / Hessenber_Matrix[j+1,j]
            
        t += 1
        Hessenber_Matrix[m+1,m] = 1
        F = scipy.linalg.expm(Hessenber_Matrix) #(m+2,m+2) matrix
        itensities_vector = beta * np.matmul(Krylov_Base, F[0:m+1, 0]) #Multiplication: number*(n_pixel,m+1)*(m+1,1) = (n_pixel,1) -> vector

        smoothed_images.append(np.reshape(itensities_vector,noisy_img.shape)) #This final image have negative intensities
        
    return smoothed_images

def main():

    lenna_img = np.asarray(Image.open('lenna.png').convert('L'))
    #lenna_face_filter = [50:225,25:200] #Image section with lenna's face
    #details_analysis_filter = [80:256,0:150]

    mean_noise = 0. #MEAN VALUE OF THE NOISE
    std_noise = 0.05 #STANDARD DEVIATION OF THE NOISE
    std_GWED = std_noise/2 * 255 #STANDARD DEVIATION OF GAUSS KERNEL
    diffusion_time = 1
    kappa_values = [0.12]#, 0.08, 0.1, 0.12, 0.14] 
    patch_sizes = [5] #SIZE OF THE PATCHES
    
    #TASK LIST TO DO, IF WANT TO DO ALL, LEAVE ALL TRUE. CANT DO 4 WITHOUT 1,2,3. CANT DO 3 WITHOUT 1,2. CANT DO 2 WITHOUT 1.
    do_noisy_image = 1 #GET THE NOISY IMAGE
    do_compressed_L = 1 #COMPUTE THE COMPRESSED LAPLACIAN TO THE ABOVE IMAGE
    do_smoothing = 1 #SMOOTH THE NOISY IMAGE WITH COMPRESSED LAPLACIAN
    do_smoothed_plots = 1 #DO THE PLOTS OF THE SMOOTHED IMAGES

    if do_noisy_image:

        print('Corruptin image additive Gaussian white noise')
        noisy_image, noisy_rmse = add_AGWN(lenna_img, mean_noise, std_noise, do_plot = False)
        
    if do_compressed_L:
            
        print('Creating the compressed Laplacian matrix')
        t2 = time.time()
        Laplacians = []
        
        for kappa in kappa_values:
            for patch_size in patch_sizes:
                Laplacians.append(Compute_Compressed_Laplacian_Matrix(noisy_image, std_GWED, kappa, patch_size))
        t3 = time.time()
        print('Took', round(t3-t2,3),'seconds to compute the Laplacian matrix')

    if do_smoothing:

        print('Starting smoothing process')
        t4 = time.time()
        
        smoothed_images = []
        for Laplacian in Laplacians:
            smoothed_images.append(arnoldi_method(- Laplacian, noisy_image, 50, diffusion_time))          
         
        t5 = time.time()
        print('Took', np.round(t5-t4,3), 'seconds to smooth the image')
        
    if do_smoothed_plots:
        
        f, axarr = plt.subplots(diffusion_time , len(kappa_values))
        
        for t in range(diffusion_time):
            for kappa_value, smoothed_img in enumerate(smoothed_images[t]):
                print('Diffusion time: %s, Kappa value: %s' %(t+1, kappa_values[kappa_value]))
                print('RMSE:', np.round(mean_squared_error(lenna_img, smoothed_img, squared = False),2))
                plt.figure()
                plt.imshow(smoothed_img, cmap = 'gray', vmin = 0, vmax = 255)
                plt.savefig('t_%s_k_%s.png' %(t+1, kappa_values[kappa_value]))
                plt.close()
                
if __name__ == "__main__":
    
    main()
