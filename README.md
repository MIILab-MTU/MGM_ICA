# Multi-graph Graph Matching for Coronary Artery Semantic Labeling in Invasive Coronary Angiograms 

## Authors
Chen Zhao<sup>1</sup>, Zhihui Xu<sup>2</sup>, Pukar Baral<sup>3</sup>, Michel Esposito<sup>4</sup>, and Weihua Zhou<sup>3,5</sup>

1. Department of Computer Science, Kennesaw State University, Marietta GA, USA 
2. Department of Cardiology, The First Affiliated Hospital of Nanjing Medical University, Nanjing, China
3. Department of Applied Computing, Michigan Technological University, Houghton, MI, USA
4. Department of Cardiology, Medical University of South Carolina, Charleston, SC, USA
5. Center for Biocomputing and Digital Health, Institute of Computing and Cyber-systems, and Health Research Institute, Michigan Technological University, Houghton, MI, USA 


## Abstract

Coronary artery disease (CAD) stands as the leading cause of death worldwide, and invasive coronary angiography (ICA) remains the gold standard for assessing vascular anatomical information. However, deep learning-based methods encounter challenges in generating semantic labels for arterial segments, primarily due to the morphological similarity between arterial branches and varying anatomy of arterial system between different projection view angles and patients. To address this challenge, we model the vascular tree as a graph and propose a multi-graph graph matching (MGM) algorithm for coronary artery semantic labeling. The MGM algorithm assesses the similarity between arterials in multiple vascular tree graphs, considering the cycle consistency between each pair of graphs. As a result, the unannotated arterial segments are appropriately labeled by matching them with annotated segments. Through the incorporation of anatomical graph structure, radiomics features, and semantic mapping, the proposed MGM model achieves an impressive accuracy of 0.9471 for coronary artery semantic labeling using our multi-site dataset with 718 ICAs. With the semantic labeled arteries, an overall accuracy of 0.9155 was achieved for stenosis detection. The proposed MGM presents a novel tool for coronary artery analysis using multiple ICA-derived graphs, offering valuable insights into vascular health and pathology. 

## Overview

![Figure 1](Figure1.png)

Figure: Workflow of multi-graph graph matching (MGM) for coronary artery semantic labeling. Top: individual graph generation; Bottom: MGM among 3 individual graphs derived from a set of ICAs. ICA<sup>1</sup> represents the ICA with unlabeled arteries, while ICA<sup>2</sup> to ICA<sup>m</sup> indicate the labeled template ICAs. MGM considers the cycle consistency between these ICAs and the relationship between arteries from different ICAs will be used to classify the unlabeled arteries in ICA<sup>1</sup>.

## Environment

`easydict==1.11`
`gurobipy==10.0.3`
`networkx==2.6.3`
`numpy==1.21.2`
`pynvml==11.5.0`
`PyYAML==6.0.2`
`scipy==1.7.3`
`torch==1.10.0`
`torch_geometric==2.0.3`
`torchvision==0.11.0`

## Example of graph matching

`python train_artery_mgm.py`

In this example, we provide the example code to show the graph matching using 3 ICA derived graphs. The output is shown below

`{'test_sample': '0', 'template_sample': '1', 'category': 'LAO_CRA', 'n': 11, 'LAD2': 'LAD2', 'D2': 'D2', 'LAD3': 'LAD3', 'LAD1': 'LAD1', 'D1': 'D1', 'LMA': 'LMA', 'LCX1': 'LCX1', 'LCX2': 'LCX2', 'OM1': 'OM1', 'OM2': 'OM2', 'LCX3': 'LCX3', 'matched': 11, 'unmatched': 0}
{'test_sample': '0', 'template_sample': '2', 'category': 'LAO_CRA', 'n': 11, 'LAD2': 'LAD2', 'D2': 'D2', 'LAD3': 'LAD3', 'LAD1': 'LAD1', 'D1': 'D1', 'LMA': 'LMA', 'LCX1': 'LCX1', 'LCX2': 'LCX2', 'OM1': 'OM1', 'OM2': 'OM2', 'LCX3': 'LCX3', 'matched': 11, 'unmatched': 0}`

