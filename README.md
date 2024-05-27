## News
* **2024/5/17** Provides a fully differentiable optimization-based approach implemented by PyTorch’s stochastic gradient descent optimizer, which is used as a baseline in our experiments. ( Reference: [Link](https://thejns.org/focus/view/journals/neurosurg-focus/54/6/article-pE16.xml) )
* **2024/1/3** Provides more similarity measures including multi-scale normalized cross-correlation, etc. ( Reference: [DiffPose](https://github.com/eigenvivek/DiffPose) )
## CMAES-reg
Using [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](https://github.com/CyberAgentAILab/cmaes) for 2D/3D registration.The procedure is  single-resolution and the DRR module is implmented by [projective spatial transformers](https://github.com/gaocong13/Projective-Spatial-Transformers).

 - The result is evaluated on simulation data. To evaluate the registration, we follow the standardized evaluation methodology to report the Mean Target Registration Error(mTRE) as follows:
<table>
    <tr>
          <td rowspan="2"> </td> <td colspan="3" align="center">mTRE(mm)</td> <td rowspan="2">SR</td>   <td rowspan="2">Reg.<br>time</td>
   </tr>
    <tr>
  		 <td align="center">95th</td> 
      	<td align="center">75th</td> 
     <td align="center">50th</td> 
    </tr>
  <tr>
     <td>Initial</td>
     <td>205.81</td>
     <td>172.05</td> 
     <td>135.12</td> 
     <td>0.0</td>
     <td>N/A</td>
    </tr>
    <tr>
     <td>CMA-ES</td>
     <td>31.72</td>
     <td>8.37</td> 
     <td>5.39</td> 
     <td>49.4</td>
     <td>21.2s</td> 
    </tr>
</table>

- The mTRE results are reported in forms of the 50th, 75th, and 95th percentiles to demonstrate the robustness of our methods. In addition, we also report the success rate (SR) and average registration time, where SR is defined as the percentage of the tested cases with a TRE smaller than 10 mm.

Although this framework is single-resolution, it can be easily changed to multi-resolution. Feel free to replace its similarity measure and tune the hyperparameters.
## Citation
If you find this work useful in your research, please cite the appropriate papers:
```
@article{chen2024optimization,
  title={An Optimization-based Baseline for Rigid 2D/3D Registration Applied to Spine Surgical Navigation Using CMA-ES},
  author={Chen, Minheng and Li, Tonglong and Zhang, Zhirun and Kong, Youyong},
  journal={arXiv preprint arXiv:2402.05642},
  year={2024}
}
```
```
@article{zhang2024introducing,
  title={Introducing Learning Rate Adaptation CMA-ES into Rigid 2D/3D Registration for Robotic Navigation in Spine Surgery},
  author={Zhang, Zhirun and Chen, Minheng},
  journal={arXiv preprint arXiv:2405.10186},
  year={2024}
}
```
