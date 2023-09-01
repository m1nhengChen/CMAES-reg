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
     <td>41.4134±70.0704</td>
     <td>95th</td> 
     <td>95th</td> 
     <td>95th</td>
     <td>N/A</td>
    </tr>
    <tr>
     <td>CMA-ES</td>
     <td>41.4134±70.0704</td>
     <td>11.3234±8.5563</td> 
     <td> 6.2626±3.3769</td> 
     <td>43.0</td>
     <td>19.3s</td> 
    </tr>
</table>

- The mTRE results are reported in forms of the 50th, 75th, and 95th percentiles to demonstrate the robustness of our methods. In addition, we also report the success rate (SR) and average registration time, where SR
is defined as the percentage of the tested cases with a TRE smaller than 10 mm.

Although this framework is single-resolution, it can be easily changed to multi-resolution. Feel free to replace its similarity measure and tune the hyperparameters.

