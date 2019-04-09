# GMM-EM_4_Multivarient
1D and 3D gaussian Mixture model

The Folder "Codes" contains the following:

Steps to run the code

1.Extract Project2_ENPM673 folder on desktop

Following is the description of the folder structure and the codes:


## Data Preparation:
	This contains:
	1. LAB images
	2. Cropped images - Green_new,Red_new,Yellow_new
	3. click.py - This is the code that was used to convert the image frames into
		LAB colour space and generate the cropped images.

## 1D_detection_gaussian:
	1. 1D.py - This code generates the 1D buoy detection video and also plots
	the gaussians of the respective buoys.
	2. EM_green.py - This generates the optimum values of means and variances
	for the Green buoy using the Green_new images using EM algo
	3. EM_red.py - This generates the optimum values of means and variances
	for the Red buoy using the Red_new images using EM algo
	4. EM_yellow_green.py - This generates the optimum values of means and variances
	for the yellow buoy using the Green_new images using EM algo
	5. EM_yellow_green.py - This generates the optimum values of means and variances
	for the yellow buoy using the Red_new images using EM algo

## 3D_Detection:
	1. 3D_detection.py - This detects the buoys using the optimum means and variances
	obtained using the Em algo.
	This has a folder Circle_detection_green in which the code Green_detection.py is
	the optimized code for detection of green channel.

## 3D_gaussian:
	Each folder contains the implementaion of EM algo for each buoy for its
	respective channel training images

	Note: All the probabilites are multiplied by the constant 100000. This is only for visualization purpose. It has anyways been normalized later.

	Link with output videos: https://drive.google.com/drive/u/0/folders/1vEHi9kKYj8Pyllbr7qTmfpCv9TX3xPRd


Thank You!!!
