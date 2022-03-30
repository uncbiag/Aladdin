# Aladdin
joint **A**t**LA**s buil**D**ing and **D**iffeomorphic reg**I**stration lear**N**ing with pairwise alignment

This is the official repository for   

**Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment**   
[Zhipeng Ding](https://biag.cs.unc.edu/author/zhipeng-ding/) and [Marc Niethammer](https://biag.cs.unc.edu/author/marc-niethammer/)   
*CVPR 2022* [eprint arxiv](https://arxiv.org/abs/2202.03563)   

If you use Aladdin or some part of the code. please cite:
```
@article{ding2022aladdin,
  title={Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment},
  author={Ding, Zhipeng and Niethammer, Marc},
  journal={arXiv preprint arXiv:2202.03563},
  year={2022}
}
```

## Key Observations
1. **(Similarity Measure)** Previous approaches for atlas building often define similarity measures between a fuzzy atlas and each individual image, which may cause alignment difficulties because a fuzzy atlas does not exhibit clear anatomical structures in contrast to the individual images. Hence, we propose pairwise image similarity loss to help the alignment. We hypothesize that incorporating a pairwise image similarity loss is beneficial for alignment accuracy.

<p align="center">
  <img src="./figs/pairwise_example_3.png" width="80%">
</p>


> In the above figure, we can clearly see that image-to-image differences are greater than atlas-to-image differences. Hence, an image-to-image similarity measure is expected to provide more alignment information than an atlas-to-image similarity measure because a fuzzy atlas does not exhibit the clear anatomic structures present in individual images. Detailed explanations can be found in the paper. 


2. **(Evaluation)** The quality of an atlas is usually evaluated in combination with the quality of the image registration algorithm. For example, an atlas framework is often evaluated based on the sharpness or entropy of the atlas, the alignment of test images in the atlas space, or the alignment of the warped atlas in test image space. These evaluation measures all have shortcomings (see the paper for details). Therefore, we propose *atlas-as-a-bridge* measure which is conceptually preferable to existing evaluation measures.

<p align="center">
  <img src="./figs/atlas_evaluation.png" width="80%">
</p>


> In the above figure, we demonstrate our proposed evaluation measure with other two commonly used measures. Evaluation on atlas space will accumulate both atlas variations and registration errors; evaluation on image space will accumulate both the estimated segmentation errors and registration errors; and using atlas-as-a-bridge will only accumulate registration errors. Hence, the proposed measure is preferable. See deatailed explanations in the paper.


3. **(Pre-processing)** Most existing atlas building methods rely on affine pre-registrations to a chosen reference image. Instead of separately considering affine and nonparametric transformations, we propose to predict a transformation which includes affine and nonparametric deformations. To achieve this goal, we used bending energy as the regularization term. We hypothesize that our combined transformation prediction is as accurate as methods that treat affine and nonparametric transformations separately.

### Pairwise Alignment
Inspired by *backward* atlas building, we propose the first pairwise alignment loss in *atlas space* as:
<p align="center">
  <img src="./figs/pairwise_loss_in_atlas_space.png" width="50%">
</p>

Inspired by *forward* atlas building, we propose the second pairwise alignment loss in *image space* as:
<p align="center">
  <img src="./figs/pairwise_loss_in_image_space.png" width="50%">
</p>

Thus, a *forward* atlas building model with pairwise image alignment can be formulated as:
<p align="center">
  <img src="./figs/forward_atlas_building_with_pairwise_alignment.png" width="50%">
</p>

This is demonstrated in the following architecture of our proposed model
<p align="center">
  <img src="./figs/architecture_comparison_3.png" width="100%">
</p>
> In the above figure, (b) is a standard forward atlas building architecture; (c) is our proposed model with pairwise image alignment losses.  Our model extends the standard model by incorporating pairwise image losses and by computing their alignments in atlas space as well as in image space using the atlas as a bridge.

### Theoretical Justification

### Evaluation Criteria

### Implementation
