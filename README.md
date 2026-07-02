# ICE原糖指数周度季节性图

这个仓库用于发布 ICE 原糖指数的三张周度季节性结构图，并展示未来 4 周多模型预测。网页默认展示 SVG 矢量图，放大缩小时会比 PNG 更清晰，同时保留 PNG 作为备用下载：

- 全样本：2004-2025 年历史 20%-80% 分位区间、中位数、上一年、当前年与回测排序最优模型预测
- 牛市条件：牛市年份样本的季节性区间
- 熊市条件：熊市年份样本的季节性区间
- 预测模型：持平基准、短期漂移、全样本季节性、状态季节性、AR 自回归、Holt 趋势、Ridge 线性回归、KNN 非线性回归与集成预测
- 交互功能：网页按滚动回测平均 MAE 排序模型，并可切换查看不同模型的未来 4 周预测值

网页入口是 `index.html`。启用 GitHub Pages 后可通过：

`https://czy123456-hub.github.io/sugar_price/`

访问。

## 目录

- `index.html`: GitHub Pages 静态网页
- `assets/`: 三张 SVG 图、三张 PNG 图、预测 CSV、回测 CSV、页面元数据
- `data/`: 原始 Excel 数据
- `src/original_matplotlib_code.py`: 用户原始 matplotlib 版本代码
- `src/generate_charts.py`: 当前仓库用于稳定生成 PNG 的 Pillow 版本脚本
- `src/forecast_models.py`: 未来 4 周多模型预测与滚动回测脚本

## 重新生成图片

```bash
python3 src/generate_charts.py
```

需要安装依赖：

```bash
pip install -r requirements.txt
```
