# Generative Adversarial Network (GAN) Architectures (Chronological Order)

## Foundational GANs
- [ ] **2014** – **InitialGAN**  
  _The original GAN introduced by Goodfellow et al., demonstrating adversarial learning between a generator and discriminator._
- [ ] **2015** – **DeepConvolutionalGAN**  
  _DCGAN: Introduced convolutional layers into GANs, significantly improving the quality and stability of image generation._

## Improved Objective Functions
- [ ] **2017** – **LeastSquareGAN**  
  _LSGAN: Replaces binary cross-entropy loss with least squares loss to stabilize training and produce higher quality outputs._
- [ ] **2017** – **WassersteinGAN**  
  _WGAN: Uses the Wasserstein distance for a more meaningful loss metric and improved training stability._
- [ ] **2017** – **WassersteinImprovedGAN**  
  _WGAN-GP: Enhances WGAN with gradient penalty to enforce the Lipschitz constraint without weight clipping._

## Conditional & Controlled Generation
- [ ] **2014** – **ConditionalGAN**  
  _cGAN: Extends GANs to generate data conditioned on labels or input features for more controlled outputs._

## Multi-Domain & Style Transfer GANs
- [ ] **2017** – **CycleGAN**  
  _Performs unpaired image-to-image translation using cycle consistency to preserve structure between domains._
- [ ] **2017** – **StarGAN**  
  _Supports image-to-image translation across multiple domains with a single model by using domain labels._
- [ ] **2018** – **StyleGAN**  
  _Introduces a style-based generator architecture for high-resolution and photorealistic image synthesis._

## Application-Specific GANs
- [ ] **2016** – **SuperResolutionGAN**  
  _SRGAN: Enhances low-resolution images with perceptual loss and adversarial training for photorealistic upscaling._

## Experimental / Theoretical Extensions
- [ ] **2020** – **DeepRegretAnalyticGAN**  
  _DRAGAN: Incorporates regret minimization for a more stable GAN training process and theoretical insights._

