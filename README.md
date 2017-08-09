## Advanced Lane Finding Project

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./example_images/Blurred_binary.png "Blurred Binary"
[image2]: ./example_images/Chess_dist_undist.png "Chessattern Undistorted"
[image3]: ./example_images/Color_combined_binary.png "Color Combined Binary"
[image4]: ./example_images/Combined_binary.png "Combined Binary"
[image5]: ./example_images/final_result.png "Final Result"
[image6]: ./example_images/HIst_Equalized.png "Hist Equ"
[image7]: ./example_images/lanes.png "Lanes"
[image8]: ./example_images/lanes_cont.png "Lanes Continued"
[image9]: ./example_images/lanes_result.png "Lanes Result"
[image10]: ./example_images/Options.png "Options"
[image11]: ./example_images/Perspective_transform.png "Perspective Transform"
[image12]: ./example_images/RL_dist_undist.png "Undistorted Test Image"

[video1]: ./output_project_video.mp4 "Video"
[video2]: ./output_challenge_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and fourth code cell of the IPython notebook located in "project_code.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]


### Pipeline (single image)

#### 1. Provide an example of a distortion-corrected image.

I used the computed coefficients and applied them on a test image. This is the result:

![alt text][image12]

As we can see, the image is now undistorted and we're ready to move on. But before we do that, I've made a quick function to undistort in code cell six using the pickled data, so we can use this without running through the other parts in the future.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The next part of my notebook is all dedicated to exploring different options and coming up with a nice binary image to calculate our curves and lanes on (up until code cell 13). 

I started with defining a bunch of functions to perform various tasks (all returning a binary image):

* Gradients
* Color Thresholds
* HLS Channel Transformations
* Equalization

I've added this last to my notebook, but saw improved results with it. I'm talking about Hist Equalize. Esepecially very bright/dark spots, such as bridges in the videos can be handled better.

The output of a test image after equalizing:

![alt text][image6]

I then went ahead and explored all of the remaining options for creating my binary image. Here is the output for each of them:

![alt text][image10]

I've decided to first create a combined binary image from the `hls_s_binary` and `hls_h_binary`. I then combined this with the `gradx` binary image ("or" combine) and finally combined this with my color thresholded binary ("and" combine). The result is a combined binary image:

![alt text][image4]

During development I wanted to be able to check out the individual contributions of the HLS and gradient binaries. That's why I created a color version (without the color_threshold applied). Note: I was able to create a way better version of this but it didn't perform well on the actual video. That's when I added the hist_equ and the color combinated one was of less use to me. Still, I like the looks of it:

![alt text][image3]

Ok cool, final step. Again, this used to be more important when I didn't have the hist_equ in place but I've kept this one because it still helps. Blurring. I applied a slight median blurr with a 3x3 kernel to the final binary image. Reduces some of the noise:

![alt text][image1]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in code cell 13. It's not a function and was split up in my final video pipeline. Please check out the notebook for details.

Anyway, I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) +55, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203.3, 720    | 320, 720      |
| 1121.6, 720   | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart (also as binary image) to verify that the lines appear parallel in the warped image.

![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

OK the next part (cells 14 - 17) are dedictaed to finding the lanes and visualizing them. First, I use a sliding window approach (initializing the values with a histogram of the bottom half of the image). I'm using 9 windows with a width margin +/- 70 pixels. There have to be at least 50pixels in the window for recentering of the window.

The sliding windows are used on the warped binary image. All found pixels within the windows are appended to left_lane_inds and right_lane_inds.

Finally, a second order polynominal is fitted to these points before vidualizing the lanes and search windows. I'm averaging the right and left lane to create a center lane. This was done during development and will not show in the final video output. The reult:

![alt text][image7]

The remaining two code cells are dedicated to finding the lanes without using the sliding window approach. This can only be done once the lanes were identified previously. I've incorporated this in my video pipeline once a lane is detected. It uses a search window from the previous values:

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated first using pixel values in cell 18. However, since we need the curvature in meters, cell 19 is where your attention should be. To display only one value I've averaged the left and right curveradii.

The position of the vehicle with respect to center is calculated in cell 20. I calculate the distance at the base of the image (also where the curvature is calculated).

In the final pipeline I calculate the radius and distance to center over an average of the last 13 frames.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

For plotting it back on our original image we need to use the inverse matrix of the one we used to warp the image. After doing that we add the lanes to our original undistorted image. This is done in code cell 21 of my notebook and produces the following result:

![alt text][image9]

Finally, I wrote the curvature and distance to center values on the image. This is my final result:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

Starting with the approach: trial and error for the most part. I've noticed that my pipeline wasn't working as expected after my first full implementation of the video pipeline. Turns out, those changes in color of the road and the concrete boundary on the left are an issue, because there is also a "line" there (between sun and shade). After I've noticed that I implemented my color thresholding (yellow and white).
However, this resulted in even more problems. It worked well for some sections, but others had faded right withe lane markings. I've added a gaussianblurr (quite high with 15x15 kernel in the beginning) to account for that. It helped but on the challenge video my problems still weren't quite solved. So I ended up with the hist_equ and a smaller gaussian blurr.

If I were to pursue this further, I'd spend more time working on the sanity checks. I did implement a couple of them (they gave me headaches at first) in the video pipeline. They check wether the two lanes cross up top, and if the distance on the bottom or top is too high (indicating non parralel lines).

TODO: Next step, since I pretty much already calculated two points (top and bottom), check for parallelism throughout and discard if not parallel. 

Averaging (I'm using deque from collections here with appendleft) over the last x frames helped a lot for smoothing out the lanes and the calculated values. The last 13 (correct) value seemed like a good number, 20 was a bit too high for my likings. The lanes were a bit slow adapting to changes.

Where my pipeline fails is the harder challenge video. Oh boy! I have to update my src and dts points for warping to begin with, account for the very curvy road and the fact that there may only be one lane detected at any given time. 1 lane detected is enough to steer the car (TESLAs are able to do so for example). If we know the width of the car we can position it accordingly.

Generally I always tested my video pipeline with all test images first (code cell 26). You can see the results [here](./output_images/)

Also, the challenge video came out alright. Not perfect. However, if you pay attention to the vehicle position calculation, you shall find that it would have been safe to steer the car with these values (maybe a bit jerky but still). This is thanks to the fact that curvature and position is evaluatet at the bottom. Even if lanes jump up top, the bottom is usually less affected.

Here's a [link to my challenge video result](./output_challenge_video.mp4)
