# GUGEN: Global User Graph Enhanced Network For Next POI Recommendation

This is a pytorch implementation of 《GUGEN: Global User Graph Enhanced Network For Next POI Recommendation》(IEEE TMC 2024, [DOI]((https://doi.org/10.1109/TMC.2024.3455107)). 

Detail information will be released after publication.


## Abstract

Learning the next Point-of-Interest (POI) is a highly context-dependent human movement behavior prediction task, which has gained increasing attention with the consideration of massive spatial-temporal trajectories data or check-in data. The spatial dependency, temporal dependency, sequential dependency and social network dependency are widely considered pivotal to predict the users’ next location in the near future. However, most existing models fail to consider the influence of other users’ movement patterns and the correlation with the POIs the user has visited. Therefore, we propose a Global User Graph Enhanced Network (GUGEN) for the next POI recommendation from a global and a user perspectives. Firstly, a trajectory learning network is designed to model the users’ short-term preference. Second, a geographical learning module is designed to
model the global and user context information. From the global perspective, two graphs are designed to represent the global POI features and the geographical relationships of all POIs. From the user perspective, a user graph is constructed to describe each users’ historical POI information. We evaluated the proposed model on three real-world datasets. The experimental evaluations demonstrate that the proposed GUGEN method outperforms the state-of-the-art approaches for the next POI recommendation.

## Performance
TABLE II: PERFORMANCE COMPARISON IN ACC@K ON THREE DATASETS
<table style="width:100%;">
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Ref</th>
    <th colspan="4">Foursquare-NYC</th>
    <th colspan="4">Foursquare-TKY</th>
    <th colspan="4">Gowalla-CA</th>
  </tr>
  <tr>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Acc@10</td>
    <td>Acc@20</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Acc@10</td>
    <td>Acc@20</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Acc@10</td>
    <td>Acc@20</td>
  </tr>
  <tr>
    <td>RNN</td>
    <td>-</td>
    <td>0.0984</td>
    <td>0.2133</td>
    <td>0.2581</td>
    <td>0.3019</td>
    <td>0.0772</td>
    <td>0.1474</td>
    <td>0.1682</td>
    <td>0.1763</td>
    <td>0.0534</td>
    <td>0.13</td>
    <td>0.1451</td>
    <td>0.1534</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>-</td>
    <td>0.0995</td>
    <td>0.2145</td>
    <td>0.2596</td>
    <td>0.304</td>
    <td>0.0778</td>
    <td>0.1476</td>
    <td>0.1691</td>
    <td>0.1765</td>
    <td>0.0535</td>
    <td>0.1302</td>
    <td>0.1461</td>
    <td>0.1543</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>-</td>
    <td>0.1018</td>
    <td>0.2174</td>
    <td>0.2607</td>
    <td>0.3057</td>
    <td>0.0782</td>
    <td>0.1493</td>
    <td>0.1697</td>
    <td>0.1771</td>
    <td>0.0541</td>
    <td>0.1306</td>
    <td>0.1470</td>
    <td>0.1548</td>
  </tr>
  <tr>
    <td>DeepMove</td>
    <td>WWW18</td>
    <td>0.1131</td>
    <td>0.2443</td>
    <td>0.2963</td>
    <td>0.3482</td>
    <td>0.0891</td>
    <td>0.1692</td>
    <td>0.1935</td>
    <td>0.2022</td>
    <td>0.0613</td>
    <td>0.1491</td>
    <td>0.1674</td>
    <td>0.1765</td>
  </tr>
  <tr>
    <td>EEDN</td>
    <td>SIGIR23</td>
    <td>0.1272</td>
    <td>0.2913</td>
    <td>0.3215</td>
    <td>0.3369</td>
    <td>0.0954</td>
    <td>0.2242</td>
    <td>0.2717</td>
    <td>0.2963</td>
    <td>0.0736</td>
    <td>0.1848</td>
    <td>0.2427</td>
    <td>0.2501</td>
  </tr>
  <tr>
    <td>LSTPM</td>
    <td>AAAI20</td>
    <td>0.2648</td>
    <td>0.5614</td>
    <td>0.6677</td>
    <td>0.7231</td>
    <td>0.2195</td>
    <td>0.3141</td>
    <td>0.4297</td>
    <td>0.5275</td>
    <td>0.1799</td>
    <td>0.3052</td>
    <td>0.3926</td>
    <td>0.4459</td>
  </tr>
  <tr>
    <td>PLSPL</td>
    <td>TKDE20</td>
    <td>0.2414</td>
    <td>0.5486</td>
    <td>0.6416</td>
    <td>0.6921</td>
    <td>0.2157</td>
    <td>0.3142</td>
    <td>0.4314</td>
    <td>0.5304</td>
    <td>0.2059</td>
    <td>0.3229</td>
    <td>0.4161</td>
    <td>0.4684</td>
  </tr>
  <tr>
    <td>STAN</td>
    <td>WWW21</td>
    <td>0.2563</td>
    <td>0.5412</td>
    <td>0.6574</td>
    <td>0.7001</td>
    <td>0.2249</td>
    <td>0.3911</td>
    <td>0.5213</td>
    <td>0.5886</td>
    <td>0.2120</td>
    <td>0.3394</td>
    <td>0.4261</td>
    <td>0.4972</td>
  </tr>
  <tr>
    <td>HiLS</td>
    <td>TMC24</td>
    <td>0.2860</td>
    <td>0.5640</td>
    <td>0.6732</td>
    <td>0.7239</td>
    <td>0.2410</td>
    <td>0.4370</td>
    <td>0.5495</td>
    <td>0.6319</td>
    <td>0.2486</td>
    <td>0.3732</td>
    <td>0.4902</td>
    <td>0.5317</td>
  </tr>
  <tr>
    <td>HGARN</td>
    <td>arXiv22</td>
    <td>0.2883</td>
    <td>0.5646</td>
    <td>0.6714</td>
    <td>0.7283</td>
    <td>0.2233</td>
    <td>0.4298</td>
    <td>0.5103</td>
    <td>0.5644</td>
    <td>0.2452</td>
    <td>0.3808</td>
    <td>0.4941</td>
    <td>0.5339</td>
  </tr>
  <tr>
    <td>AGRAN</td>
    <td>SIGIR23</td>
    <td>0.2134</td>
    <td>0.5194</td>
    <td>0.5997</td>
    <td>0.6382</td>
    <td>0.2302</td>
    <td>0.4265</td>
    <td>0.5258</td>
    <td>0.5967</td>
    <td>0.2502</td>
    <td>0.3975</td>
    <td>0.5088</td>
    <td>0.5386</td>
  </tr>
  <tr>
    <td>GETNext</td>
    <td>SIGIR22</td>
    <td>0.2792</td>
    <td>0.5611</td>
    <td>0.6733</td>
    <td>0.7396</td>
    <td>0.2539</td>
    <td>0.4523</td>
    <td>0.5613</td>
    <td>0.6504</td>
    <td>0.2484</td>
    <td>0.4312</td>
    <td>0.4981</td>
    <td>0.5668</td>
  </tr>
  <tr>
    <td>FPGT</td>
    <td>ASC23</td>
    <td>0.2814</td>
    <td>0.5630</td>
    <td>0.6771</td>
    <td>0.7494</td>
    <td>0.2603</td>
    <td>0.4712</td>
    <td>0.5765</td>
    <td>0.6586</td>
    <td>0.2647</td>
    <td>0.4382</td>
    <td>0.5153</td>
    <td>0.5737</td>
  </tr>
  <tr>
    <td>SNPM</td>
    <td>AAAI23</td>
    <td>0.2775</td>
    <td>0.5638</td>
    <td>0.6814</td>
    <td><b>0.7813</b></td>
    <td>0.2604</td>
    <td>0.5095</td>
    <td>0.5872</td>
    <td><b>0.7134</b></td>
    <td>0.2714</td>
    <td>0.4389</td>
    <td>0.5171</td>
    <td>0.5747</td>
  </tr>
  <tr>
    <td>GUGEN</td>
    <td></td>
    <td><b>0.2981</b></td>
    <td><b>0.5925</b></td>
    <td><b>0.6929</b></td>
    <td>0.7566</td>
    <td><b>0.2743</b></td>
    <td><b>0.5132</b></td>
    <td><b>0.5916</b></td>
    <td>0.6631</td>
    <td><b>0.2857</b></td>
    <td>0.4532</b></td>
    <td><b>0.5250</b></td>
    <td><b>0.5874</b></td>
  </tr>
</table>

TABLE III: PERFORMANCE COMPARISON IN MAP@K AND MRR ON THREE DATASETS
<table style="width:100%;">
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Ref</th>
    <th colspan="4">Foursquare-NYC</th>
    <th colspan="4">Foursquare-TKY</th>
    <th colspan="4">Gowalla-CA</th>
  </tr>
  <tr>
    <td>MAP@5</td>
    <td>MAP@10</td>
    <td>MAP@20</td>
    <td>MRR</td>
    <td>MAP@5</td>
    <td>MAP@10</td>
    <td>MAP@20</td>
    <td>MRR</td>
    <td>MAP@5</td>
    <td>MAP@10</td>
    <td>MAP@20</td>
    <td>MRR</td>
  </tr>
  <tr>
    <td>RNN</td>
    <td>-</td>
    <td>0.1383</td>
    <td>0.1449</td>
    <td>0.1477</td>
    <td>0.1514</td>
    <td>0.1272</td>
    <td>0.1333</td>
    <td>0.1378</td>
    <td>0.1410</td>
    <td>0.0968</td>
    <td>0.1042</td>
    <td>0.1077</td>
    <td>0.1114</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>-</td>
    <td>0.1402</td>
    <td>0.1461</td>
    <td>0.1496</td>
    <td>0.1526</td>
    <td>0.1282</td>
    <td>0.1352</td>
    <td>0.1389</td>
    <td>0.1428</td>
    <td>0.0982</td>
    <td>0.1051</td>
    <td>0.1086</td>
    <td>0.1128</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>-</td>
    <td>0.1451</td>
    <td>0.1520</td>
    <td>0.1545</td>
    <td>0.1576</td>
    <td>0.1334</td>
    <td>0.1399</td>
    <td>0.1434</td>
    <td>0.1475</td>
    <td>0.1016</td>
    <td>0.1085</td>
    <td>0.1130</td>
    <td>0.1165</td>
  </tr>
  <tr>
    <td>DeepMove</td>
    <td>WWW18</td>
    <td>0.1615</td>
    <td>0.1684</td>
    <td>0.1721</td>
    <td>0.1758</td>
    <td>0.1478</td>
    <td>0.1557</td>
    <td>0.1600</td>
    <td>0.1643</td>
    <td>0.1131</td>
    <td>0.1210</td>
    <td>0.1252</td>
    <td>0.1298</td>
  </tr>
  <tr>
    <td>EEDN</td>
    <td>SIGIR23</td>
    <td>0.1868</td>
    <td>0.1911</td>
    <td>0.2014</td>
    <td>0.2023</td>
    <td>0.1643</td>
    <td>0.1826</td>
    <td>0.1921</td>
    <td>0.1941</td>
    <td>0.1425</td>
    <td>0.1941</td>
    <td>0.2065</td>
    <td>0.2072</td>
  </tr>
  <tr>
    <td>LSTPM</td>
    <td>AAAI20</td>
    <td>0.3601</td>
    <td>0.3747</td>
    <td>0.3802</td>
    <td>0.3828</td>
    <td>0.3026</td>
    <td>0.3142</td>
    <td>0.3196</td>
    <td>0.3241</td>
    <td>0.2424</td>
    <td>0.2500</td>
    <td>0.2536</td>
    <td>0.2565</td>
  </tr>
  <tr>
    <td>PLSPL</td>
    <td>TKDE20</td>
    <td>0.2936</td>
    <td>0.3386</td>
    <td>0.3725</td>
    <td>0.3784</td>
    <td>0.2139</td>
    <td>0.2906</td>
    <td>0.3335</td>
    <td>0.3343</td>
    <td>0.2721</td>
    <td>0.2868</td>
    <td>0.2934</td>
    <td>0.2941</td>
  </tr>
  <tr>
    <td>STAN</td>
    <td>WWW21</td>
    <td>0.3342</td>
    <td>0.3732</td>
    <td>0.3894</td>
    <td>0.3901</td>
    <td>0.2743</td>
    <td>0.3322</td>
    <td>0.3526</td>
    <td>0.3535</td>
    <td>0.2862</td>
    <td>0.2938</td>
    <td>0.3073</td>
    <td>0.3104</td>
  </tr>
  <tr>
    <td>HiLS</td>
    <td>TMC24</td>
    <td>0.3904</td>
    <td>0.4132</td>
    <td>0.4182</td>
    <td>0.4194</td>
    <td>0.3415</td>
    <td>0.3601</td>
    <td>0.3702</td>
    <td>0.3734</td>
    <td>0.2894</td>
    <td>0.3054</td>
    <td>0.3086</td>
    <td>0.3109</td>
  </tr>
  <tr>
    <td>HGARN</td>
    <td>arXiv22</td>
    <td>0.3945</td>
    <td>0.4155</td>
    <td>0.4178</td>
    <td>0.4219</td>
    <td>0.3168</td>
    <td>0.3292</td>
    <td>0.3349</td>
    <td>0.3385</td>
    <td>0.2951</td>
    <td>0.3003</td>
    <td>0.3118</td>
    <td>0.3146</td>
  </tr>
  <tr>
    <td>AGRAN</td>
    <td>SIGIR23</td>
    <td>0.3431</td>
    <td>0.3584</td>
    <td>0.3715</td>
    <td>0.3721</td>
    <td>0.3263</td>
    <td>0.3344</td>
    <td>0.3584</td>
    <td>0.3606</td>
    <td>0.2988</td>
    <td>0.3077</td>
    <td>0.3120</td>
    <td>0.3167</td>
  </tr>
  <tr>
    <td>GETNext</td>
    <td>SIGIR22</td>
    <td>0.3854</td>
    <td>0.4026</td>
    <td>0.4086</td>
    <td>0.4113</td>
    <td>0.3541</td>
    <td>0.3682</td>
    <td>0.3732</td>
    <td>0.3764</td>
    <td>0.3103</td>
    <td>0.3239</td>
    <td>0.3311</td>
    <td>0.3372</td>
  </tr>
  <tr>
    <td>FPGT</td>
    <td>ASC23</td>
    <td>0.3793</td>
    <td>0.4041</td>
    <td>0.4112</td>
    <td>0.4136</td>
    <td>0.3622</td>
    <td>0.3758</td>
    <td>0.3792</td>
    <td>0.3837</td>
    <td>0.3245</td>
    <td>0.3469</td>
    <td>0.3470</td>
    <td>0.3485</td>
  </tr>
  <tr>
    <td>SNPM</td>
    <td>AAAI23</td>
    <td>0.3721</td>
    <td>0.4172</td>
    <td><b>0.4413</b></td>
    <td><b>0.4428</b></td>
    <td>0.3716</td>
    <td>0.3786</td>
    <td><b>0.4153</b></td>
    <td><b>0.4161</b></td>
    <td>0.3202</td>
    <td>0.3443</td>
    <td>0.3487</td>
    <td>0.3506</td>
  </tr>
  <tr>
    <td>GUGEN</td>
    <td></td>
    <td><b>0.4092</b></td>
    <td><b>0.4226</b></td>
    <td>0.4276</td>
    <td>0.4296</td>
    <td><b>0.3736</b></td>
    <td><b>0.3842</b></td>
    <td>0.3892</td>
    <td>0.3939</td>
    <td><b>0.3329</b></td>
    <td><b>0.3662</b></td>
    <td><b>0.3695</b></td>
    <td><b>0.3723</b></td>
  </tr>
</table>

## Architecture

```
<TBD>
```

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/GUGEN
```

## Project Structure

- train_model.py: The training code of our model
- utils.py: Dataset code
- Model.py: main file for Model
- requirements.txt: List the pip dependencies

## Dependency

After installing the dependency:

    pip install -r requirements.txt

## Train

- Download datasets from [here](https://drive.google.com/drive/folders/1o72mNxUgSJX43KcQ2Tg_YE3N3oJEZX5T?usp=drive_link).
- Run `train_model.py`.


## Citation

```
<TBD>
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
