# ⚡ Odd-One-Out: Anomaly Detection by Comparing with Neighbors 

Goal: Detecting 'odd-looking' samples in multi-object scene environments. 

<img src=figures/o3.gif>

<table>
  <tr>
      <strong><a href="https://arxiv.org/abs/xxxxxx">Odd-One-Out: Anomaly Detection by Comparing with Neighbors</a></strong><br>
      Ankan Bhunia, Changjian Li, Hakan Bilen<br>
  </tr>
</table>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://openaccess.thecvf.com/content/CVPR2024/papers/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.pdf)
[![dataset](https://img.shields.io/badge/Dataset-link-blue)]()


## 🌀 Multi-object AD vs previous works

<table>
    <tr>
    <td>(a) Standard AD setting</td>
    <td>The input is usually an image of single object. In this setting,
      <ul>
        <li>Anomalies are instance-specific.</li> 
        <li>This setting suffers from <b>ambiguous</b> definition of regularity (e.g., both cups in the example can be either normal or anomalous based on what the manufacturer wants to produce).</li>
        <li>To resolve the ambiguity an explicit specification of regularity is required to provide (e.g., <a href="https://github.com/VICO-UoE/Looking3D">Looking3D</a>).</li>
      </ul>
    </td>
    <td><img src="figures/SAD.jpg" width="800"/></td>
    </tr>
    <tr>
      <td>(b) Proposed Multi-object AD setting</td>
      <td>The input of the framework is a set of sparse view images of a scene containing multiple objects. We aim to detect 'odd-looking' objects that contain manufacturing errors (e.g., different geometry, texture) or damages (e.g., cracks, fractures).
      <ul>
        <li>Anomalies are scene-specific.</li>
        <li>Explicit specification of regularity is not required when seeing objects in group. Majority of objects are normal, which act as regularity reference. </li>
        <li>This is useful in quality control in bulk production.</li>
      </ul>
      </td>
       <td><img src="figures/MAD.jpg" width="800"/></td>
    </tr>
</table>

## 🏙️ More examples of our proposed dataset


<table>
  <tr>
    <td><img src="figures/sample_1.gif" width="200"/></td>
    <td><img src="figures/sample_2.gif" width="200"/></td>
    <td><img src="figures/sample_3.gif" width="200"/></td>
  </tr>
  <tr>
    <td><img src="figures/sample_4.gif" width="200"/></td>
    <td><img src="figures/sample_5.gif" width="200"/></td>
    <td><img src="figures/sample_6.gif" width="200"/></td>
  </tr>
</table>





# Codes and Dataset coming soon! 
