<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/femiogundare/prostate-cancer-genomics">
    <img src="readme_images/nn.png" alt="Logo" width="500" height="270">
  </a>

<h2 align="center">Q-ProstateNet</h2>

  <p align="center">
    <h4 align="center">Q-ProstateNet: A Biologically Informed Graph Neural Network for Personalized Treatment of Prostate Cancer</h4>
    <br />
    <br />
    <br />
    <a href="https://htmlpreview.github.io/?https://github.com/femiogundare/prostate-cancer-genomics/blob/main/_plots/figure3/sankey_diagram.html" align="center">View interactive network architecture</a>
    ·
    ·
  </p>
</p>

<hr>

<p>This repository is my implementation of the paper <b><a href="https://www.nature.com/articles/s41586-021-03922-4">Biologically Informed Deep Neural Network for Prostate Cancer Discovery</a></b> by Haitham Elmarakeby, Eliezer Van Allen, et al.</p>
<p>Haitham Elmarakeby is currently an Instructor at the Dana-Farber Cancer Institute in Boston, Massachusetts. Eliezer Van Allen is an associate member at the Broad Institute of MIT and Harvard and an Assistant Professor at the Dana-Farber Cancer Institute and Harvard Medical School. You can read about the Van Allen Lab <a href="https://vanallenlab.dana-farber.org/">here</a>.</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
    </li>
    <li><a href="#network-architecture">Neural Network Architecture</a></li>
    <li><a href="#train-test">Training and Testing</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#findings">Findings</a></li>
    <li><a href="#supplementary">Supplementary Figures</a></li>
    <li><a href="#References">References</a></li>
  </ol>
</details>


<!-- about-the-project -->
## About the Project
<p>Here, I developed a deep learning predictive model that can differentiate between the genomic profiles of prostate cancers that are lethal and those that are very much unlikely to cause symptoms or death. The model, called Q-ProstateNet, may help physicians know in advance whether a prostate cancer patient's tumor will spread to other parts of the body or become more resistant to treatment over time. Q-ProstateNet can also identify molecular features, genes, and biological pathways that may be linked to disease progression.</p>


<!-- network-architecture -->
## Neural Network Architecture
<p>A set of 3,007 curated biological pathways were used to build Q-ProstateNet. In Q-ProstateNet, the molecular profile of an individual is fed into the model and distributed over a layer of nodes representing a set of genes using weighted links.</p>

<b>Fig. 1 | Neural Network Architecture</b>. Q-ProstateNet encodes various biological entities into a neural network with customized connections between consecutive layers (that is, features from patient profile, genes, pathways, biological processes and outcome).
<p>
<img src="readme_images/femi.jpg" alt="arch" width="500" height="400">
</p>
<br />

<b>Fig. 2 | Inspection and interpretation of Q-ProstateNet</b>. Visualization of the inner layers of Q-ProstateNet shows the approximate relative importance of different nodes in each layer. Nodes on the far left denote feature types; the nodes in the second layer denote genes; the next layers denote higher-level biological entities; and the final layer denote the model outcome.
<p align="center">
<img src="_plots/figure3/sankey_diagram.png" alt="node rankings" width="1000" height="500">
</p>
<br />


<!-- train-test -->
## Training and Testing
<p>Q-ProstateNet was trained on data such as genomic sequences and somatic, or uninherited, mutations from more than 1,000 prostate cancer patients. On testing the model on data from other prostate cancer patients, it was found that it correctly distinguishes about 80% of metastatic tumors from primary, less advanced tumors.</p>


<!-- results -->
## Results
<p>The trained Q-ProstateNet outperformed traditional machine learning algorithms, including logistic regression, decision trees, and linear and radial basis support vector machines (area under the receiver operating characteristic (ROC) curve (AUC) = 0.93, area under the precision-recall curve (AUPRC) = 0.89, accuracy = 0.84).</p>

Table | Diagnostic performance of Q-ProstateNet compared to traditional machine learning algorithms.

| Model | AUC | Precision | Recall | f1 | AUPRC | AUPRC |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Q-ProstateNet | 0.93 | 0.72 | 0.82 | 0.77 | 0.89 | 0.84 |
| Logistic Regression | 0.91 | 0.81 | 0.64 | 0.72 | 0.84 | 0.83 |
| Linear SVM | 0.91 | 0.80 | 0.61 | 0.69 | 0.85 | 0.82 |
| RBF SVM | 0.91 | 0.82 | 0.61 | 0.70 | 0.86 | 0.83 |
| Decision Tree | 0.83 | 0.88 | 0.63 | 0.73 | 0.73 | 0.85 |
| Random Forest | 0.88 | 0.75 | 0.57 | 0.64 | 0.78 | 0.79 |
| AdaBoost | 0.89 | 0.93 | 0.57 | 0.70 | 0.83 | 0.84 |


<b>Fig. 3 | Receiver operating curves of Q-ProstateNet and traditional machine learning alorithms</b>.
<p>
<img src="_plots/figure1/auc.png" alt="auc" width="500" height="400">
</p>
<br />
<b>Fig. 4 | Precision-recall curves of Q-ProstateNet and traditional machine learning alorithms</b>.
<p>
<img src="_plots/figure1/prc.png" alt="prc" width="500" height="400">
</p>
<br />
<b>Fig. 5 | Confusion matrix of Q-prostateNet</b>.
<p>
<img src="_plots/figure1/confusion matrix.png" alt="confusion matrix" width="500" height="400">
</p>
<br />
<b>Fig. 6 | Performance comparision of Q-ProstateNet (P-Net) and a dense fully connected network</b>.
<p>
<img src="_plots/figure2/pnet_vs_densenet/pnet_vs_dense_sameweights_auc.png" alt="performance comparision" width="500" height="400">
</p>
<br />

<!-- findings -->
## Findings 
<p>It was found that among aggregate molecular alterations, copy number variation was more informative compared with mutations. This is in agreement with the findings of Hieronymus, et al. in their paper <b><a href="https://pubmed.ncbi.nlm.nih.gov/25024180/">Copy number alteration burden predicts prostate cancer relapse</a></b>.</p>

<b>Fig. 7 | Joint distribution of AR, TP53 and MDM4 alterations across 1,013 prostate cancer samples using an UpSetPlot</b>. A gene is said to be altered if it has a mutation, deep deletion or high amplification.
<p>
<img src="_plots/figure4/joint_distribution_of_AR_TP53_MDM4.png" alt="joint" width="500" height="400">
</p>
<br />

When MDM4 was turned off using gene editing (CRISPR-Cas9), data analysis showed that cell proliferation decreased, implying that the cancer cells could be more sensitive to treatment. This suggests two things:
* Popular prostate cancer medications like Xtandi® (enzalutamide) and Zytiga® (abiraterone acetate), which are antiandrogenic in action, may be less effective in the treatment of prostate cancer patients whose genomic profiles show an overexpression of MDM4;
* Big biopharmaceutical companies could remodel drugs that inhibit MDM4 to treat prostate tumors.
<p>
<img src="readme_images/CRISPR.png" alt="crispr" width="500" height="400">
</p>
<br />


<!-- supplementary -->
## Supplementary Figures
<b>Fig. 8 | Graph of mutation against copy-number alteration</b>.
<p>
<img src="_plots/analysis/cnv_mutation.png" alt="cnv_mut" width="500" height="400">
</p>
<br />


<b>Fig. 9 | Relative ranking of nodes in each layer of Q-ProstateNet</b>.
<p>
<img src="_plots/extended_figures/node_rankings.png" alt="node rankings" width="1000" height="1000">
</p>
<br />


<b>Fig. 10 | Activation distribution of important nodes in each layer of Q-ProstateNet</b>.
<p>
<img src="_plots/extended_figures/activation_distribution.png" alt="activations" width="1000" height="1000">
</p>
<br />


<!-- References -->
## References
* Elmarakeby H, et al. "Biologically informed deep neural network for prostate cancer classification and discovery." Nature. Online September 22, 2021. DOI: 10.1038/s41586-021-03922-4
* Armenia, Joshua, et al. "The long tail of oncogenic drivers in prostate cancer." Nature genetics 50.5 (2018): 645-651.
* Robinson, Dan R., et al. "Integrative clinical genomics of metastatic cancer." Nature 548.7667 (2017): 297-303.
* Fraser, Michael, et al. "Genomic hallmarks of localized, non-indolent prostate cancer." Nature 541.7637 (2017): 359-364.
* Abida, W., et al. "Genomic correlates of clinical outcome in advanced prostate cancer." Proc. Natl Acad. Sci. USA 116, 11428–11436 (2019).
* Fabregat, A., et al. "The Reactome pathway knowledgebase." Nucleic Acids Res. 46, D649–D655 (2018).
* Hieronymus, H., et al. "Copy number alteration burden predicts prostate cancer relapse." Proc. Natl Acad. Sci. USA 111, 11139–11144 (2014).
