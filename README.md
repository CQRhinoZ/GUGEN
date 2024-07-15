# GUGEN
# GUGEN: Global User Graph Enhanced Network For Next POI Recommendation

This is a pytorch implementation of 《GUGEN: Global User Graph Enhanced Network For Next POI Recommendation》(IEEE TMC 2024, under review). 


## Abstract

Learning the next Point-of-Interest (POI) is a highly context-dependent human movement behavior prediction task, which has gained increasing attention with the consideration of massive spatial-temporal trajectories data or check-in data. The spatial dependency, temporal dependency, sequential dependency and social network dependency are widely considered pivotal to predict the users’ next location in the near future. However, most existing models fail to consider the influence of other users’ movement patterns and the correlation with the POIs the user has visited. Therefore, we propose a Global User Graph Enhanced Network (GUGEN) for the next POI recommendation from a global and a user perspectives. Firstly, a trajectory learning network is designed to model the users’ short-term preference. Second, a geographical learning module is designed to
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
    <td>0.212</td>
    <td>0.3394</td>
    <td>0.4261</td>
    <td>0.4972</td>
  </tr>
  <tr>
    <td>HiLS</td>
    <td>TMC24</td>
    <td>0.286</td>
    <td>0.564</td>
    <td>0.6732</td>
    <td>0.7239</td>
    <td>0.241</td>
    <td>0.437</td>
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
    <td>0.563</td>
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
    <td><b>0.5925</b</td>
    <td><b>0.6929</b</td>
    <td>0.7566</td>
    <td><b>0.2743</b</td>
    <td><b>0.5132</b</td>
    <td><b>0.5916</b</td>
    <td>0.6631</td>
    <td><b>0.2857</b</td>
    <td>0.4532</b</td>
    <td><b>0.525</b</td>
    <td><b>0.5874</b</td>
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
    <td>0.212</td>
    <td>0.3394</td>
    <td>0.4261</td>
    <td>0.4972</td>
  </tr>
  <tr>
    <td>HiLS</td>
    <td>TMC24</td>
    <td>0.286</td>
    <td>0.564</td>
    <td>0.6732</td>
    <td>0.7239</td>
    <td>0.241</td>
    <td>0.437</td>
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
    <td>0.563</td>
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
    <td><b>0.5925</b</td>
    <td><b>0.6929</b</td>
    <td>0.7566</td>
    <td><b>0.2743</b</td>
    <td><b>0.5132</b</td>
    <td><b>0.5916</b</td>
    <td>0.6631</td>
    <td><b>0.2857</b</td>
    <td>0.4532</b</td>
    <td><b>0.525</b</td>
    <td><b>0.5874</b</td>
  </tr>
</table>

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
GUGEN: Global User Graph Enhanced Network For Next POI Recommendation, IEEE TMC 2024 (under review)
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
