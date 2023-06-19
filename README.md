# Continuous Layout Editing of Single Images with Diffusion Models

![alt text](figures/teaser.jpg)

Zhiyuan Zhang $^{1*}$, Zhitong Huang $^{1*}$, [Jing Liao](https://liaojing.github.io/html/) $^{1\dagger}$

<font size="1"> $^1$: City University of Hong Kong, Hong Kong SAR
<font size="1"> $^*$: Both authors contributed equally to this research &nbsp;&nbsp; $^\dagger$: Corresponding author </font>

## Abstract:
Recent advancements in large-scale text-to-image diffusion models have enabled many applications in image editing. However, none of these methods have been able to edit the layout of single existing images. To address this gap, we propose the first framework for layout editing of a single image while preserving its visual properties, thus allowing for continuous editing on a single image. Our approach is achieved through two key modules. First, to preserve the characteristics of multiple objects within an image, we disentangle the concepts of different objects and embed them into separate textual tokens using a novel method called masked textual inversion. Next, we propose a training-free optimization method to perform layout control for a pre-trained diffusion model, which allows us to regenerate images with learned concepts and align them with user-specified layouts. As the first framework to edit the layout of existing images, we demonstrate that our method is effective and outperforms other baselines that were modified to support this task. Our code will be freely available for public use upon acceptance.
