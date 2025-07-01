# 3D-SPARK
3D Spatial Assay for Replication Kinetics: segmentation and analysis pipelines

This repository contains analysis pipelines for manuscript "" (https://www.biorxiv.org/content/10.1101/2025.02.21.639251v1)

Contents of the repository:
- QuantSIM: Python module folder containing all custom-made functions for the segmentation, overlap assessment and visualization of 3D-SPARK processed files
- 01_segmentation_pipeline_examples: Jupyter notebooks as exemplary pipelines to run segmentation of 3D-SIM stacks following 3D-SPARK protocol.
- 02_segmentation_analysis_pipelines: R markdown scripts as exemplary pipelines to summarise different output results of 01_segmentation_pipeline_examples.

List of Python packages used for analysis is in "requirements.txt" file.
List of R packages used for analysis appear in each R markdown script.

Input for 01_segmentation_pipeline_examples (information of how these files have to be organized is in every notebook):
- 3D-SIM stack (czi format, by default searches for "_Out.czi" files)
- Nuclei files:
-   3D-SIM DAPI slices or stacks (will only identify nuclei on maximum Z projection)
-   Mask .tiff files

Output from 01_segmentation_pipeline_examples:
- Segmentation of each channel, overlaps between channels, non-overlapping, initiation, ongoing and termination events, stored as compressed numpy arrays in "layers" and "overlap_layers" folders.
- Quantified segmented regions (channel label, area, spatial location, intensity, other skimage features extractable), stored as csv files in "tables".
- Quantified overlaps and event types (overlap label as composition of two channel labels, each channel label involved in overlap or event), stored as csv files in "overlap_tables".
- Graphs showcasing distributation of volumes and distances of all regions, non-overlapping, initiation, ongoing or termination events, stored as png files.
- Key parameters of the segmentation analysis run, stored as "000 SIM_segmentation extra info"
- Summary of overlap analysis performed, to quickly see quantified overlap numbers and events, stored as"000 SIM_analysis CHANNEL1_vs_CHANNEL2 extra info.txt"

Input for 02_segmentation_analysis_pipelines:
- Output from 01_segmentation_pipeline_examples (depending on script, uses different files)

Output from 02_segmentation_analysis_pipelines (all is within the R markdown script, can export plots with "ggsave" function from "ggplot2" package):
- Summary of each event type per analyzed nucleus (__event_types)
- Summary of ongoing event volumes, ratio of overlap against whole volume, and distances between centroids (__overlap_stats)
- Summary of distance of each segmented region or each event to nuclear border or nuclear centroid (__periphery)
- Summary of overlap of each channel compared to each segmented region volume (__segmented_areas_ratios)
- Summary of distance of each region or event to SON foci (__SON_domains)
- Summary of all segmented regions per channel (__segmented_numbers)


Analysis pipelines were developed by Bruno Urién González, PhD student in Bennie Lemmens group at SciLifeLab/Karolinska Institutet since 2022.
