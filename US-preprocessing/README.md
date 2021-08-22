# US-preprocessing

Ultrasound data preprocessing

![Generating m-mode images](generated_gifs/mmode_generation.gif) 

Here is python code for doing a few things:
- Getting a mask for the actual ultrasound image from the video view sent by an ultrasound probe
- Create a mapping from curvilinear to rectilinear ultrasound images
- Use that mapping to resample images (to rectilinear)
  - This was re-implemented as an external ITK module in C++
- Generate an m-mode US image from a b-mode video and an m-mode line index

TODO
- Generate m-mode image from ROI box
