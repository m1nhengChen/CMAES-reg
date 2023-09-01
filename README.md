# CMAES-reg
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
     <td>205.81±89.39</td>
     <td>172.05±66.78</td> 
     <td>135.12±48.70</td> 
     <td>0.0</td>
     <td>N/A</td>
    </tr>
    <tr>
     <td>CMA-ES</td>
     <td>31.72±62.13</td>
     <td>8.37±4.95</td> 
     <td>5.39±2.40</td> 
     <td>49.4</td>
     <td>21.2s</td> 
    </tr>
</table>

- The mTRE results are reported in forms of the 50th, 75th, and 95th percentiles to demonstrate the robustness of our methods. In addition, we also report the success rate (SR) and average registration time, where SR
is defined as the percentage of the tested cases with a TRE smaller than 10 mm.

Although this framework is single-resolution, it can be easily changed to multi-resolution. Feel free to replace its similarity measure and tune the hyperparameters.

