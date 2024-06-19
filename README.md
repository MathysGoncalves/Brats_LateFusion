
# Brain Tumor Segmentation using Multi-modal MRI and Late Fusion Techniques

## Overview
Brain tumors are among the deadliest types of cancer. Specifically, glioblastoma, and diffuse astrocytic glioma with molecular features of glioblastoma (WHO Grade 4 astrocytoma), are the most common and aggressive malignant primary tumor of the central nervous system in adults, with extreme intrinsic heterogeneity in appearance, shape, and histology, with a median survival of approximately 15 months. Brain tumors in general are challenging to diagnose, hard to treat and inherently resistant to conventional therapy because of the challenges in delivering drugs to the brain, as well as the inherent high heterogeneity of these tumors in their radiographic, morphologic, and molecular landscapes. Years of extensive research to improve diagnosis, characterization, and treatment have decreased mortality rates in the U.S by 7% over the past 30 years.

Only a notebook is presented because .py formats are more difficult to handle on platforms offering GPU.

## Data Description 
The Brain Tumor Segmentation (BraTS) Continuous Challenge seeks to identify the current, state-of-the-art segmentation algorithms for brain diffuse glioma patients and their sub-regions.
The BraTS training and validation data available for download and methodological development by the participating teams describe a total of 5,880 MRI scans from 1,470 brain diffuse glioma patients and are identical to the data curated for the RSNA-ASNR-MICCAI BraTS 2021 Challenge.
All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions.

The data is organized in the following structure:

├───Brats_Data\
│   ├──BraTS-GLI-00100-000\
│   │   ├──BraTS-GLI-00100-000-t2w.nii.gz\
│   │   ├──BraTS-GLI-00100-000-t2f.nii.gz\
│   │   ├──BraTS-GLI-00100-000-t1n.nii.gz\
│   │   ├──BraTS-GLI-00100-000-t1c.nii.gz\
│   │   └──BraTS-GLI-00100-000-seg.nii.gz\
│   ├──BraTS-GLI-00099-000\
│   │   └──BraTS-GLI-00099-000-ft2w.nii.gz\
│   └──...



## Preprocessing
1.	**Loading the Data**: The MRI volumes and their corresponding segmentation labels were loaded using nibabel, a Python package for reading neuroimaging data.

2. **Data Augmentation**: Given the limited size of the dataset, data augmentation was applied to increase the variability and robustness of the training data. The augmentations included:
- Random Zoom: With a probability of 0.15, a zoom factor was sampled uniformly from (1.0, 1.4) and applied to the input volume using cubic interpolation, while the labels were interpolated using nearest neighbor.
- Random Flips: With a probability of 0.5, each volume was randomly flipped along the x, y, and z axes independently.
- Gaussian Noise: With a probability of 0.15, random Gaussian noise with mean zero and standard deviation sampled uniformly from (0, 0.33) was added to the input volume.
- Gaussian Blur: With a probability of 0.15, Gaussian blurring with a standard deviation sampled uniformly from (0.5, 1.5) was applied.
- Brightness Adjustment: With a probability of 0.15, a random value was sampled uniformly from (0.7, 1.3) and used to scale the voxel intensities.
- Contrast Adjustment: With a probability of 0.15, the contrast was adjusted by scaling the voxel intensities with a random value sampled uniformly from (0.65, 1.5).

3.	**Normalization**: Each MRI volume was normalized to have zero mean and unit variance. This step is crucial for ensuring that the intensities are within a similar range across different patients and modalities.

4.	**Resampling and Cropping**: The MRI volumes were resampled to a common voxel spacing and cropped to focus on the brain region, removing unnecessary background and reducing the computational load.
5.	**Transformation Pipeline**: We used TorchIO, a Python library specifically designed for medical imaging preprocessing, to implement the transformations and augmentations in a reproducible and efficient manner.

### Labels
Annotations comprise the GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), as described in the latest BraTS summarizing paper.
Instead of classes present in the labels the BraTS leaderboard is computed based on three partially overlapping regions: 
-	Whole tumor (1, 2, 3)
-	Tumor core (1, 3) 
-	Enhancing tumor (3)


## Model Architecture
In this project, we utilized a 3D U-Net architecture for brain tumor segmentation. The U-Net architecture is well-suited for biomedical image segmentation tasks due to its ability to capture both local and global context through a series of convolutional and upsampling layers. Using 3D convolutions allows the model to learn from volumetric data directly, capturing spatial dependencies across slices which is crucial for accurate segmentation of brain tumors.

### U-Net Architecture
The U-Net model consists of two main parts:
1.	Encoder (Contracting Path): This part captures the context in the image through a series of convolutional layers followed by max-pooling operations, which progressively reduce the spatial dimensions while increasing the depth of feature maps.
2.	Decoder (Expanding Path): This part reconstructs the image by upsampling the feature maps and concatenating them with corresponding feature maps from the contracting path to recover spatial information lost during downsampling.
Architectural Details
- Double Convolution Blocks: Each block in the U-Net consists of two 3D convolutional layers, each followed by a batch normalization layer and a ReLU activation function. This combination helps in learning complex features while maintaining the stability and efficiency of training.
- Downsampling: The downsampling is performed using 3D max-pooling layers, which reduce the spatial dimensions by a factor of 2.
- Upsampling: The upsampling is done using transposed convolutions (also known as deconvolutions), which increase the spatial dimensions by a factor of 2. This is followed by concatenation with corresponding feature maps from the contracting path to retain spatial information.
- Final Convolution: The final layer is a 1x1x1 convolution that reduces the number of channels to the number of target classes (in this case, three: Complete, Core, and Enhancing).

### Loss function
We utilized the Dice coefficient as the primary loss function for training our 3D U-Net model. The Dice coefficient is a popular choice for medical image segmentation tasks due to its ability to directly measure the overlap between predicted and true segmentation masks.

The Dice coefficient is defined as: 
Dice = 2 * |A ∩ B| / (|A| + |B|)

Where A is the predicted segmentation and B is the ground truth segmentation. The Dice coefficient ranges from 0 to 1, where 1 indicates perfect overlap.

Dice Loss is derived from the Dice coefficient as follows:
Dice Loss = 1 - Dice


This loss function is particularly advantageous for several reasons:
- **Direct Optimization for Overlap**: The Dice coefficient directly measures the overlap between the predicted and true segmentation, which is the ultimate goal in segmentation tasks. By optimizing the Dice loss, we are explicitly training the model to maximize this overlap.
- **Handling Class Imbalance**: In medical image segmentation, the target regions (e.g., tumors) often occupy a small portion of the image compared to the background. The Dice loss is less sensitive to class imbalance because it focuses on the overlap of the segmented regions rather than the absolute number of correctly classified pixels.
- **Smooth Differentiability**: The Dice loss is differentiable, which is essential for gradient-based optimization methods used in training deep neural networks.

#### Alternative Loss function
The Dice loss was chosen for this project due to its effectiveness in directly optimizing for segmentation overlap and handling class imbalance. However, exploring alternative loss functions like Tversky loss, focal loss, and combined Dice-BCE loss can provide additional insights and potentially improve model performance in different scenarios.

**Tversky loss** is a generalization of the Dice loss that introduces additional parameters to control the trade-off between false positives and false negatives.
- More flexible than Dice loss for handling different types of class imbalance.
- Allows customization based on specific application needs.
- Requires tuning of additional hyperparameters (α and β).

**Focal loss** is designed to address the issue of class imbalance by focusing more on hard-to-classify examples.
- Effective for highly imbalanced datasets.
- Reduces the impact of easy-to-classify examples, focusing training on hard examples.
- Requires tuning of the focusing parameter 𝛾.

A **combination of Dice loss and BCE** loss can be used to leverage the strengths of both loss functions.
- Balances the overlap-focused optimization of Dice loss with the probabilistic output of BCE loss.
- Can provide better performance in certain scenarios.
- Requires tuning of the weighting factor 𝜆.


## Memory Optimization Methods
Training 3D convolutional neural networks, especially on large medical image datasets, can be computationally expensive and memory-intensive. To mitigate these challenges, several memory optimization techniques were employed:

- Mixed precision training leverages both 16-bit (half-precision) and 32-bit (single-precision) floating-point arithmetic to reduce memory usage and increase computational speed. This was implemented using NVIDIA's Automatic Mixed Precision (AMP) library in PyTorch, which automatically scales the model to use the appropriate precision without significant changes to the code.

- Gradient accumulation allows the effective batch size to be increased without requiring more GPU memory by accumulating gradients over multiple mini-batches before performing a backward pass. This technique was particularly useful when the hardware constraints limited the maximum batch size.

- Gradient checkpointing trades computation for memory by saving only some of the intermediate activations and recomputing the others during the backward pass. This significantly reduces memory usage at the cost of increased computation time.


## Fusion Methods 
We employed various fusion methods to combine the segmentation results from different MRI modalities. Fusion methods play a crucial role in leveraging the complementary information present in different modalities, leading to improved segmentation performance. Below are the detailed descriptions and implementations of the fusion methods used:

- **Average Fusion**: computes the voxel-wise average of the predictions from each modality. This method assumes that each modality contributes equally to the final segmentation result. Simple and effective, assumes equal contribution from each modality.

- **Majority voting** (hard voting): involves taking a voxel-wise majority vote among the predictions. The final segmentation is determined by the class that has the majority of votes across all modalities. Robust to outliers, good for binary predictions.

- **Maximum fusion**: selects the maximum value across the predictions from different modalities for each voxel. This method highlights the most confident predictions among the modalities. Highlights confident predictions, may ignore subtle information.

- **Soft voting**, similar to average fusion, computes the voxel-wise average of the predictions from each modality, but uses the raw probabilities instead of hard labels. This method is particularly useful when combining probabilistic outputs.

- **Weighted average**: computes the weighted average of the predictions from different modalities. This method allows for assigning different importance to each modality based on their relevance or performance. Allows assigning importance to each modality, requires weight tuning.

- **Fusion Network**: involves using a small neural network to learn how to combine the predictions from different modalities. This method allows for learning more complex relationships and dependencies between the modalities. Learns complex relationships, requires additional training. Combines probabilistic outputs, useful for soft predictions.


## Evaluation Process
As requested by the challenge, we will use the following metrics for model evaluation:

**Dice Score**
The Dice Score measures the overlap between the predicted segmentation and the ground truth segmentation. Provided a measure of overlap between the predicted and true segmentations, directly reflecting the segmentation accuracy.

**Hausdorff Distance (95th Percentile):**
The Hausdorff Distance measures the maximum distance between the predicted and ground truth surfaces. The 95th percentile of the Hausdorff Distance is often used to reduce the influence of outliers. Measured the worst-case surface distance, indicating how well the model predicted the boundaries of the tumors.

### Results 

First, the models were evaluated separately for each target, to see which models performed best for each target. Then they were evaluated on each target after the fusion.

The models were evaluated on a batch size of 2, due to the heavy weight of the data on a total of 10 epochs and a sample of only 50 images.
Cross validation was implemented but not used, as it took too long to train.
Training for the four modalities took around 8 hours on a 15GB T4 GPU.

Taking this information into account, the results are in no way representative of reality on the ground.

| Modality        | Dice Whole Tumor | Dice Tumor Core | Dice Enhancing Tumor |
|-----------------|------------------|-----------------|----------------------|
| T1n              | 0.20             | 0.10            | 0.05                 |
| T1c             | 0.25             | 0.12            | 0.08                 |
| T2w              | 0.18             | 0.09            | 0.04                 |
| T2f           | 0.22             | 0.11            | 0.06                 |
| Average         | 0.21             | 0.10            | 0.06                 |
| Majority Voting | 0.20             | 0.11            | 0.07                 |
| Fusion Network  | 0.23             | 0.12            | 0.08                 |
| Maximum         | 0.19             | 0.13            | 0.09                 |               
| Soft Voting     | 0.21             | 0.11            | 0.07                 |
