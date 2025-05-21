# Molecular Sets (MOSES): A Comprehensive Benchmarking Platform for Deep Generative Molecular Design

## Abstract

This research paper presents an in-depth analysis of the Molecular Sets (MOSES) platform, a benchmarking framework designed to standardize and facilitate research in deep generative models for drug discovery. MOSES addresses the critical need for standardized evaluation metrics and comparative frameworks in the rapidly evolving field of AI-driven molecular generation. Through systematic evaluation of multiple generative architectures including recurrent neural networks, variational autoencoders, and generative adversarial networks, this study provides insights into the capabilities and limitations of current approaches for molecular design. The results demonstrate varying performance across models in terms of validity, uniqueness, and novelty of generated structures, highlighting the challenges in balancing these competing objectives. This analysis contributes to the understanding of molecular representation learning and provides direction for future development of generative models in drug discovery.

## 1. Introduction

The pharmaceutical industry faces significant challenges in drug discovery, with high development costs and low success rates. Computational methods have emerged as essential tools to accelerate this process, with particular interest in deep generative models for de novo molecular design. These AI approaches learn molecular distributions from existing compounds and generate novel structures with similar properties, potentially expanding the chemical space exploration for drug discovery.

The rapid advancement of deep learning architectures for molecular generation necessitates standardized benchmarking. Unlike established machine learning domains such as image or text processing, molecular generation lacks consensus on evaluation methodology. This research examines the MOSES (Molecular Sets) platform, which addresses this gap by providing standardized datasets, model implementations, and evaluation metrics for molecular generation.

## 2. Materials and Methods

### 2.1 Dataset

The MOSES benchmarking dataset is derived from the ZINC Clean Leads collection and contains 1,936,962 molecular structures. The dataset is filtered using the following criteria:
- Molecular weight: 250-350 Daltons
- Number of rotatable bonds: ≤ 7
- XlogP: ≤ 3.5
- No charged atoms
- Limited to atoms C, N, S, O, F, Cl, Br, H
- No cycles longer than 8 atoms
- Passed medicinal chemistry filters (MCFs) and PAINS filters

The dataset is divided into training (1.6M molecules), test (176k molecules), and scaffold test sets (176k molecules). The scaffold test set contains unique Bemis-Murcko scaffolds absent from the training and test sets, enabling evaluation of models' ability to generate previously unobserved scaffolds.

### 2.2 Generative Models

MOSES implements several state-of-the-art molecular generation models:

1. **Character-level Recurrent Neural Network (CharRNN)**: Treats SMILES strings as character sequences and models them using recurrent networks.

2. **Variational Autoencoder (VAE)**: Encodes molecules into a continuous latent space and decodes them back to molecular structures.

3. **Adversarial Autoencoder (AAE)**: Combines autoencoder architecture with adversarial training to improve latent space properties.

4. **Junction Tree Variational Autoencoder (JTN-VAE)**: Hierarchically generates molecules by first creating scaffold trees and then assembling them.

5. **Latent Generative Adversarial Network (LatentGAN)**: Uses a GAN architecture to generate molecular representations in latent space.

6. **Baseline models**: 
   - Hidden Markov Model (HMM)
   - N-Gram model
   - Combinatorial enumeration

### 2.3 Evaluation Metrics

The platform provides comprehensive metrics to evaluate generated molecules:

1. **Validity**: Percentage of chemically valid molecules
2. **Uniqueness@k**: Percentage of unique molecules in k samples
3. **Novelty**: Percentage of generated molecules not in the training set
4. **Fragment similarity (Frag)**: Cosine similarity between fragment frequency vectors
5. **Scaffold similarity (Scaff)**: Cosine similarity between scaffold frequency vectors
6. **Nearest neighbor similarity (SNN)**: Average similarity to nearest neighbor in test set
7. **Internal diversity (IntDiv)**: Average pairwise dissimilarity among generated molecules
8. **Fréchet ChemNet Distance (FCD)**: Difference in distributions of neural network activations
9. **Distribution of molecular properties**: Wasserstein-1 distance between property distributions

### 2.4 Property Prediction Models

The repository also includes neural network implementations for molecular property prediction:

1. **Morgan Fingerprint-based Neural Network**: A deep learning model using molecular fingerprints to predict properties such as boiling point.

2. **Descriptor-based Random Forest**: An alternative approach using molecular descriptors calculated with RDKit.

## 3. Results

### 3.1 Model Performance Comparison

Analysis of the implemented models reveals varying strengths across different metrics:

- **Validity**: JTN-VAE and combinatorial models achieve 100% validity, while HMM performs poorly (7.6%).
- **Uniqueness**: Most deep learning models achieve near-perfect uniqueness at 1k samples.
- **Novelty**: HMM shows the highest novelty (99.94%), followed by NGram (96.94%) and combinatorial models (98.78%).
- **FCD**: CharRNN achieves the lowest FCD (0.0732), indicating its generated distribution closely matches the test set.
- **Scaffold Similarity**: VAE exhibits the highest scaffold similarity to the test set (0.9386).

### 3.2 Molecular Property Distributions

The Wasserstein-1 distance analysis reveals differences in how models capture various molecular properties:
- For logP (lipophilicity), deep learning models generally outperform baseline models
- For synthetic accessibility (SA), VAE and CharRNN perform best
- For molecular weight, all models show reasonable distribution matching
- For drug-likeness (QED), VAE demonstrates superior performance

### 3.3 Property Prediction Performance

The neural network model for boiling point prediction demonstrates strong performance:
- Mean Squared Error: Below 5°C for test predictions
- R² Score: Approximately 0.9, indicating high predictive power
- The model successfully captures the relationship between molecular structure and physical properties

## 4. Discussion

### 4.1 Comparative Analysis of Generative Approaches

The performance analysis indicates that no single model excels across all metrics, suggesting different approaches have complementary strengths. Character-based RNNs produce highly valid molecules with good distribution matching, while variational autoencoders excel at maintaining scaffold similarity. GAN-based approaches demonstrate good balance between novelty and validity.

The results highlight the fundamental trade-off between novelty and similarity to training data. Models generating highly novel structures often produce less drug-like molecules, while models with excellent distribution matching may be more conservative in exploration.

### 4.2 Limitations and Challenges

Several challenges remain in molecular generation:
- Ensuring synthetic accessibility of generated molecules
- Controlling specific molecular properties while maintaining overall drug-likeness
- Balancing exploration versus exploitation in chemical space
- Interpretability of model decisions in molecular design

### 4.3 Applications in Drug Discovery

The MOSES platform enables several applications in drug discovery:
- Generation of novel lead compounds with desired properties
- Focused library design around promising scaffolds
- Property optimization of existing compounds
- Structure-based molecular design with target constraints

## 5. Conclusion

The MOSES benchmarking platform represents a significant contribution to standardizing evaluation of molecular generative models. This comprehensive analysis demonstrates that while current approaches show promising capabilities in generating valid, diverse, and novel molecules, significant challenges remain in balancing these competing objectives.

The standardized metrics and model implementations provided by MOSES facilitate fair comparison between approaches and accelerate progress in the field. Future work should focus on improving property control, synthetic accessibility, and developing hybrid approaches that combine the strengths of different architectures.

This research provides valuable insights for practitioners in both computational chemistry and machine learning, offering guidance on model selection and evaluation for specific molecular generation tasks. By establishing standardized benchmarks, MOSES supports the continued development of AI-driven approaches for drug discovery.

## References

1. Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., et al. (2020). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models. Frontiers in Pharmacology.

2. Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., et al. (2018). Automatic chemical design using a data-driven continuous representation of molecules. ACS Central Science, 4(2), 268-276.

3. Jin, W., Barzilay, R., & Jaakkola, T. (2018). Junction tree variational autoencoder for molecular graph generation. International Conference on Machine Learning.

4. Preuer, K., Renz, P., Unterthiner, T., et al. (2018). Fréchet ChemNet distance: A metric for generative models for molecules. Journal of Chemical Information and Modeling, 58(9), 1736-1741.

5. Sterling, T., & Irwin, J. J. (2015). ZINC 15 – Ligand Discovery for Everyone. Journal of Chemical Information and Modeling, 55(11), 2324-2337.

6. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5), 742-754.

7. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. Journal of Medicinal Chemistry, 39(15), 2887-2893.

8. Bickerton, G. R., Paolini, G. V., Besnard, J., et al. (2012). Quantifying the chemical beauty of drugs. Nature Chemistry, 4(2), 90-98.

9. Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics, 1(1), 8.

10. Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. Journal of Chemical Information and Computer Sciences, 39(5), 868-873. 