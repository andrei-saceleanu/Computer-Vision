# Homework 2 Computer Vision
#### Saceleanu Andrei-Iulian, IA1-B

### Task 1

- results/task1

- as far as implementation is concerned, the own convolutional filtering
is shown to be within a tolerance to OpenCV's filter2D (see test mode in ex1.py)
with cv2.BORDER_CONSTANT set on.

    - for faster results, filter2D was used from there onwards

#### Gaussian filter

- when keeping same image(i.e. same scale) and same sigma value,
larger kernel size translates into a brighter image
(the filter will contain more positive values and increase the overall pixel value)

- when keeping same image and same kernel size,
larger sigma translates into a darker image
(the filter values are same in number across the different cases, but smaller in magnitude)
and a larger bluriness

- when keeping same sigma value and same kernel size,
the perceived bluriness is smaller for larger resolution images
(the pixel dimensions of various objects/structures in the larger images
are correspondingly larger, making the influence of a fixed kernel smaller)

#### Box filter

- when keeping same image, larger kernel size translates into a larger perceived bluriness
(with more prominent edge effects accordingly)

- when keeping same kernel size, the larger images have smaller perceived bluriness

#### Filter relation

- the 2 images from results/task1/single_experiments seem to confirm the relation between box filter width
and gaussian filter sigma (https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf)

    - $ \sigma = \sqrt{\frac{w^2-1}{12}} $


### Task 2

- results/task2

- while the clarity and confidence(i.e. pixel value) of detected edges vary across resolutions,
the majority of them(in various sizes) can be found at different scales, with the following mentions:

    - subtle edges in the high-res 100 image are not as prominent in 010 for example

    - in lower resolutions, fragmented or discontinous edges are more common because of the limited
connectivity options(for Canny detector in particular, this may be due to less accurate gradient information)

    - edges in lower resolution appear less distinct, thicker than in higher resolution

### Task 3

- results/task3

#### Alternative 1

- ex3.py

- each object was isolated in the initial image by selecting the appropriate channels
for the requested colors (by indexing, could have been performed with a 3D kernel otherwise)
and performing thresholding with trial and error values that give good results on all scales
(often except 0025)
    - e.g. for the blue pool, 3D kernel could have had all zero values, except a middle 1 in Blue
corresponding dimension
    - while the blue pool had an almost pure blue RGB code, the other objects required handling at least 2 channels to be correctly identified

- in order to enhance the connectedness of the objects in the thresholded image and remove random small structures, morphological operations have been applied(i.e. dilation = max filter, erosion = min filter)

#### Alternative 2

- ex3_sol2.py

- assuming known the color content and the pixel dimensions of the object, the detection could be performed
using a template matching style approach(https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html)
    - e.g. for the blue pool, the center pixel will have maximum value in the conv map obtained
with a filter of all ones (the values were scaled to [0-1] )

- the solution works by scaling the dimensions extracted from the full resolution image(except 0025)

- similar results could have been obtained for the other 2 objects