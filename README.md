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
     <td>205.8090±89.3876</td>
     <td>172.0497±66.7819</td> 
     <td>135.1181±48.6957</td> 
     <td>95th</td>
     <td>N/A</td>
    </tr>
    <tr>
     <td>CMA-ES</td>
     <td>31.7168±62.1311</td>
     <td>8.3734±4.9519</td> 
     <td>5.3905±2.3998</td> 
     <td>49.4</td>
     <td>21.2s</td> 
    </tr>
</table>

- The mTRE results are reported in forms of the 50th, 75th, and 95th percentiles to demonstrate the robustness of our methods. In addition, we also report the success rate (SR) and average registration time, where SR
is defined as the percentage of the tested cases with a TRE smaller than 10 mm.

Although this framework is single-resolution, it can be easily changed to multi-resolution. Feel free to replace its similarity measure and tune the hyperparameters.

