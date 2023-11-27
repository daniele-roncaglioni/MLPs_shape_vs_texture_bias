# Resources

https://github.com/gregorbachmann/scaling_mlps
https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW

# Usage

This repo is a fork of https://github.com/gregorbachmann/scaling_mlps
We also have a mirror at https://github.com/daniele-roncaglioni/scaling_mlps_mirror


DA...data augmentations (flips, very light crops, rotations)

https://cs.stanford.edu/~acoates/stl10/
STL10: 96x96, 10 classes,  13k labelled images

https://github.com/fastai/imagenette
Imagenette2: original sizes or 320x320 or 160x160, 1o classes, 13.4k images


Core Experiments:
Goal: test performance of MLPs on shape vs texture in very low resolution regime (harder to learn texture)
(B_6-W_512)x(Imagenette2-64, Imagenette2-64-Stylized)x(DA)
(B_6-W_512, B_12-W_1024) x (Imagenette2-160, Imagenette2-160-Stylized) x (DA)

