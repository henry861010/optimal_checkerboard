# feature_lines 演算法資料流說明

這份文件說明 `src/optimal_checkerboard/algorithms/feature_lines.py` 相關的資料流、程式呼叫順序、各 function 的責任，以及每個階段產生的資料格式。這段流程的目的，是把輸入的幾何 face 轉成 checkerboard mesh 需要的共享 grid rail 與 snap rule；後段也說明 `OptimalMesh25D.apply_snap_rules_at_z()` 如何用這些 snap rules 修正每個 z layer 的節點。

## 核心目的

`feature_lines` 流程做三件事：

1. 從 `BOX`、`POLYGON`、`LINE` face 抽出 xy 線段，並附上 active z interval。
2. 將線段分成 x rail 與 y rail 候選特徵。
3. 將彼此很接近、且不會造成同一 z 平面 snap 衝突的線段合併成 shared rail，並產生每個 z layer 的 snap rules。

在整個 mesher 流程裡，它位於前處理階段：

```text
OptimalMesh25D.set_pattern()
  -> _get_feature_lines()
      -> _extract_lines()
          -> _box_to_lines()
      -> build_shared_rails()
          -> _classify_line()
          -> _line_to_feature()
          -> _build_axis_rails()
              -> _can_add_to_rail()
                  -> _has_near_endpoint_conflict()
                  -> _has_blocking_intermediate_feature()
              -> _new_rail()
              -> _add_to_rail()
              -> _serialize_rail()
              -> _rail_snap_rules()
OptimalMesh25D.mesh_checkerboard_box()
  -> _build_rail_node_index()
      -> _build_node_axis_rail_ids()
OptimalMesh25D.apply_snap_rules_at_z(z)
  -> _rules_by_axis_and_rail()
  -> _snap_rule_node_ids()
      -> _span_node_ids()
      -> _coupled_corner_node_ids()
```

## 對外呼叫入口

主要入口在 `OptimalMesh25D.set_pattern()`：

```python
merge_tol = ratio * element_size
_get_feature_lines(faces, merge_tol, return_details=True)
```

這裡要注意命名：`_get_feature_lines(faces, element_size, ...)` 的第二個參數在 function docstring 裡叫 `element_size`，但 `set_pattern()` 實際傳入的是 `merge_tol`。也就是說，在目前程式裡 `_get_feature_lines()` 收到的值代表「允許合併 shared rail 的最大座標距離」，不是原始 mesh element size。

## 輸入資料格式

`faces` 是 face dictionary list，每個 dictionary 有 `type`、`dim`、`bottom_z`、`top_z`。`bottom_z` / `top_z` 表示這個 face 在 z 方向的 active interval；feature line 會在 `bottom_z` 產生 snap event，並保留 `top_z` 供 rail 合併判斷 active z overlap。

### BOX

```python
{
    "type": "BOX",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

`_box_to_lines()` 把 box 的 xy 矩形轉成四條邊，並在每條線上附上 `[bottom_z, top_z]`。

輸出線段：

```python
[
    [[x1, y1], [x2, y1], [z0, z1]],
    [[x2, y1], [x2, y2], [z0, z1]],
    [[x2, y2], [x1, y2], [z0, z1]],
    [[x1, y2], [x1, y1], [z0, z1]],
]
```

### LINE

```python
{
    "type": "LINE",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

`_extract_lines()` 會把 flat 2D dim 轉成 canonical line：

```python
[[x1, y1], [x2, y2], [z0, z1]]
```

後續 `_classify_line()` 會檢查線段必須是水平或垂直。

### POLYGON

```python
{
    "type": "POLYGON",
    "dim": [[x0, y0], [x1, y1], ...],
    "bottom_z": z0,
    "top_z": z1,
}
```

`dim` 是一個封閉 orthogonal polygon 的 point list。程式用 `[poly[i - 1], point]` 建立邊，所以第一個點會自動連回最後一個點。

## 第一階段：抽線段

### `_get_feature_lines(faces, element_size, return_details=False)`

位置：`feature_lines.py`

這是 `feature_lines` 的主函式。

流程：

1. 呼叫 `_extract_lines(faces)`，把 face 轉成 raw lines。
2. 呼叫 `build_shared_rails(lines, element_size)`，把 raw lines 轉成 shared rail metadata。
3. 從 rail metadata 整理舊版 API 相容的回傳值。
4. 如果 `return_details=True`，額外回傳 snap rules 與完整 rail metadata。

基本回傳格式：

```python
(
    group_lines_v,
    group_lines_h,
    x_list,
    y_list,
)
```

詳細回傳格式：

```python
(
    group_lines_v,
    group_lines_h,
    x_list,
    y_list,
    snap_rules_by_z,
    rails,
)
```

其中：

```python
group_lines_v = [rail["lines_by_z"] for rail in x_rails]
group_lines_h = [rail["lines_by_z"] for rail in y_rails]
x_list = [x_rail["coord"], ...]
y_list = [y_rail["coord"], ...]
rails = {"x": x_rails, "y": y_rails}
```

`group_lines_v` 的每個 item 是一條 x rail 對應的原始垂直線段，依 z 分組。`group_lines_h` 則是一條 y rail 對應的原始水平線段，依 z 分組。

範例：

```python
[
    {
        0.0: [
            [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
            [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
        ]
    }
]
```

### `_extract_lines(faces)`

位置：`feature_lines.py`

責任：把不同 face type 統一轉成 raw line list。

輸入：

```python
faces = [
    {
        "type": "BOX",
        "dim": [0, 0, 10, 5],
        "bottom_z": 0,
        "top_z": 3,
    },
    {
        "type": "LINE",
        "dim": [20, 1, 25, 1],
        "bottom_z": 0,
        "top_z": 0,
    },
]
```

輸出：

```python
lines = [
    [[0, 0], [10, 0], [0, 3]],
    [[10, 0], [10, 5], [0, 3]],
    [[10, 5], [0, 5], [0, 3]],
    [[0, 5], [0, 0], [0, 3]],
    [[20, 1], [25, 1], [0, 0]],
]
```

若遇到不支援的 `type`，會丟出：

```python
ValueError("Unsupported face type: ...")
```

### `_box_to_lines(dim, z_range)`

位置：`feature_lines.py`

責任：把一個矩形 box face 轉成四條水平或垂直邊。

重要細節：

- `dim` 必須是 `[x1, y1, x2, y2]`。
- `z_range` 必須是 `[bottom_z, top_z]`。
- 回傳順序是下、右、上、左四條邊。
- 沒有檢查 box 是否真的 axis-aligned，後續分類階段會拒絕非水平或非垂直線段。

## 第二階段：分類與標準化

### `build_shared_rails(lines, merge_tol, eps=0.01, z_decimals=4)`

位置：`rail_builder.py`

責任：把 raw lines 轉成 shared rails、rail coordinate list，以及依 z 分組的 snap rules。

流程：

1. `_classify_line(lines, eps)` 將線段分成 vertical 與 horizontal。
2. vertical lines 轉成 axis=`"x"` 的 feature。
3. horizontal lines 轉成 axis=`"y"` 的 feature。
4. `_build_axis_rails()` 分別建立 x rails 與 y rails。
5. 合併 x/y snap rules，依 z 分組成 `snap_rules_by_z`。

回傳格式：

```python
{
    "x_rails": [...],
    "y_rails": [...],
    "x_list": [x_rail_coord, ...],
    "y_list": [y_rail_coord, ...],
    "snap_rules_by_z": {
        z_value: [snap_rule, ...],
        ...
    },
}
```

### `rail_builder.py` 的角色總覽

`rail_builder.py` 是整個演算法真正決定「哪些 pattern lines 可以共用同一條 checkerboard rail」的地方。`feature_lines.py` 主要負責把 face 轉成線段並整理回傳格式；合併規則、snap rule 建立、rail metadata 序列化都在 `rail_builder.py`。

資料在 `rail_builder.py` 裡會經過四種形態：

```text
raw lines
  -> classified lines
  -> normalized features
  -> internal rails
  -> public rails + snap rules
```

對應 function：

| 階段 | Function | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 分類 | `_classify_line()` | raw lines | `vertical_lines`, `horizontal_lines` |
| 標準化 | `_line_to_feature()` | 單條 line | feature dict |
| 建 rail | `_build_axis_rails()` | 同 axis features | public rails, snap rules |
| 合併判斷 | `_can_add_to_rail()` | rail + feature | `True` / `False` |
| 更新 rail | `_new_rail()` / `_add_to_rail()` | feature 或 rail + feature | internal rail |
| 輸出 rail | `_serialize_rail()` | internal rail | public rail dict |
| 輸出 snap | `_rail_snap_rules()` | internal rail | snap rule list |

`build_shared_rails()` 是這些步驟的 orchestration function。它本身不直接做複雜幾何判斷，而是把 x/y 兩個方向拆開，分別交給 `_build_axis_rails()`。

### `build_shared_rails()` 詳細資料流

`build_shared_rails(lines, merge_tol, eps=0.01, z_decimals=4)` 收到的是 `_extract_lines()` 產生的 raw line list：

```python
lines = [
    [[x1, y1], [x2, y2], [z_bottom, z_top]],
    ...
]
```

第一步先分類：

```python
vertical_lines, horizontal_lines = _classify_line(lines, eps=eps)
```

接著轉成 features。這裡 `feature_id` 是在各自方向內重新從 0 開始編號，所以 x feature 的 `feature_id=0` 和 y feature 的 `feature_id=0` 可以同時存在，因為它們會在不同 axis pipeline 裡處理。

```python
vertical_features = [
    _line_to_feature(line, "x", feature_id, z_decimals=z_decimals)
    for feature_id, line in enumerate(vertical_lines)
]

horizontal_features = [
    _line_to_feature(line, "y", feature_id, z_decimals=z_decimals)
    for feature_id, line in enumerate(horizontal_lines)
]
```

然後分別建立 rails：

```python
x_rails, x_snap_rules = _build_axis_rails(
    vertical_features,
    "x",
    merge_tol,
    eps,
)

y_rails, y_snap_rules = _build_axis_rails(
    horizontal_features,
    "y",
    merge_tol,
    eps,
)
```

最後將 x/y snap rules 合併，依 z 分組：

```python
snap_rules_by_z = defaultdict(list)
for rule in x_snap_rules + y_snap_rules:
    snap_rules_by_z[rule["z"]].append(rule)
```

再轉成一般 dict，並排序：

```python
snap_rules_by_z = {
    z: sorted(
        rules,
        key=lambda rule: (
            rule["axis"],
            rule["rail_id"],
            rule["span_min"],
        ),
    )
    for z, rules in sorted(snap_rules_by_z.items(), key=lambda item: item[0])
}
```

最終回傳：

```python
{
    "x_rails": x_rails,
    "y_rails": y_rails,
    "x_list": [rail["coord"] for rail in x_rails],
    "y_list": [rail["coord"] for rail in y_rails],
    "snap_rules_by_z": snap_rules_by_z,
}
```

這裡的 `x_list` / `y_list` 是 checkerboard mesh 需要放進 grid 的 shared rail coordinates，不一定等於原始 pattern line 的真實座標。真實座標會保留在 snap rules 的 `target_coord` 裡。

### `_classify_line(lines, eps=0.01)`

位置：`classify_line.py`

責任：檢查 raw line 合法性，並依方向分類。

合法條件：

- canonical line 的 z interval 必須滿足 `z_bottom <= z_top`。
- legacy `[[x1, y1, z], [x2, y2, z]]` line 仍可被 `_classify_line()` / `build_shared_rails()` 直接處理，但兩端 z 必須相同。
- vertical line：`abs(x1 - x2) <= eps` 且 `abs(y1 - y2) > eps`。
- horizontal line：`abs(y1 - y2) <= eps` 且 `abs(x1 - x2) > eps`。
- diagonal line 與 zero-length line 都會被拒絕。

回傳格式：

```python
(
    vertical_lines,
    horizontal_lines,
)
```

排序規則：

- vertical lines 依 `(z, x)` 排序。
- horizontal lines 依 `(z, y)` 排序。

在這個專案的命名裡：

- vertical line 表示 x 固定、y 變動，因此會形成 x rail。
- horizontal line 表示 y 固定、x 變動，因此會形成 y rail。

### `_line_to_feature(line, axis, feature_id, z_decimals=4)`

位置：`rail_builder.py`

責任：把幾何線段轉成後續合併 rail 方便使用的標準 feature dictionary。

vertical line 轉成 x feature：

```python
line = [[1, 0], [1, 10], [0, 20]]
feature = {
    "feature_id": 0,
    "axis": "x",
    "coord": 1.0,
    "z": 0.0,
    "z_bottom": 0.0,
    "z_top": 20.0,
    "span_min": 0.0,
    "span_max": 10.0,
    "line": [[1.0, 0.0], [1.0, 10.0], [0.0, 20.0]],
}
```

horizontal line 轉成 y feature：

```python
line = [[0, 3], [5, 3], [0, 20]]
feature = {
    "feature_id": 0,
    "axis": "y",
    "coord": 3.0,
    "z": 0.0,
    "z_bottom": 0.0,
    "z_top": 20.0,
    "span_min": 0.0,
    "span_max": 5.0,
    "line": [[0.0, 3.0], [5.0, 3.0], [0.0, 20.0]],
}
```

欄位意義：

| 欄位 | 意義 |
| --- | --- |
| `feature_id` | 該方向內的線段 id，用於避開自己與 deduplicate |
| `axis` | `"x"` 表示 x rail，`"y"` 表示 y rail |
| `coord` | 此線段固定的座標，x rail 用 x，y rail 用 y |
| `z` | snap rule 生效的 z layer，等於 `z_bottom` 並預設 round 到 4 位小數 |
| `z_bottom` / `z_top` | feature active z interval，用於判斷不同 face 是否會在 z 方向 overlap |
| `span_min` / `span_max` | 線段在另一個軸向上的覆蓋範圍 |
| `line` | float 化後的原始線段 |

## 第三階段：建立 shared rail

### `_build_axis_rails(features, axis, merge_tol, eps)`

位置：`rail_builder.py`

責任：針對同一個 axis，把可以共用 checkerboard grid line 的 features 合併成 shared rails，並產生 snap rules。

輸入：

```python
features = [feature, ...]
axis = "x" or "y"
merge_tol = 允許合併的最大座標差
eps = 浮點容忍值
```

演算法流程：

1. 依 `(coord, z_bottom, z_top, span_min)` 排序 feature。
2. 用 active rail window 跳過已經離目前 feature 太遠的 rails。
3. 逐一處理 feature。
4. 對候選 rails 呼叫 `_can_add_to_rail()` 判斷是否能合併。
5. 若有多個可合併 rail，選擇與目前 feature 距離最近的 rail。
6. 無法合併就 `_new_rail()`，可以合併就 `_add_to_rail()`。
7. 最後依 rail coord 排序，序列化成 public rail，並產生 snap rules。

更接近程式碼的流程如下：

```python
if not features:
    return [], []

sorted_features = sorted(
    features,
    key=lambda item: (
        item["coord"],
        item["z_bottom"],
        item["z_top"],
        item["span_min"],
    ),
)
rails = []
active_start = 0

for feature in sorted_features:
    # 1. 移動 active_start，排除已經不可能合併的舊 rails。
    while (
        active_start < len(rails)
        and feature["coord"] - rails[active_start]["max_coord"]
        > merge_tol + eps
    ):
        active_start += 1

    # 2. 在 active rails 裡找可以合併且距離最近的 rail。
    best_index = None
    best_distance = None
    for index in range(active_start, len(rails)):
        rail = rails[index]
        if feature["coord"] < rail["min_coord"] - merge_tol - eps:
            continue
        if not _can_add_to_rail(rail, feature, merge_tol, eps, features):
            continue

        distance = abs(feature["coord"] - rail["coord"])
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance

    # 3. 找不到 rail 就新建，找得到就加入最佳 rail。
    if best_index is None:
        rails.append(_new_rail(axis, feature))
    else:
        _add_to_rail(rails[best_index], feature)
```

### `active_start` 的意義

`sorted_features` 是依 `coord` 由小到大掃描，因此當目前 feature 的座標已經比某條 rail 的 `max_coord` 大超過 `merge_tol + eps` 時，後面的 features 只會更大，不可能再跟那條 rail 合併。

所以程式用 `active_start` 當作候選 rail 起點：

```python
feature["coord"] - rails[active_start]["max_coord"] > merge_tol + eps
```

只要這個條件成立，該 rail 就可以從候選集合排除。這是一個簡單的 sliding-window optimization，避免每個 feature 都掃描所有舊 rails。

### 為什麼選距離最近的 rail

同一個 feature 可能同時能加入多條 rail。程式用以下距離挑出最佳 rail：

```python
distance = abs(feature["coord"] - rail["coord"])
```

選距離最近者可以讓 shared rail 的 grouping 更局部，避免 feature 被吸收到比較遠但仍合法的 rail，造成後續 rail coord 中點偏移過大。

### `_build_axis_rails()` 的輸出

`_build_axis_rails()` 回傳兩個 list：

```python
(
    public_rails,
    snap_rules,
)
```

`public_rails` 已經是 `_serialize_rail()` 後的格式，不包含 `members` 與 `spans_by_z`。

`snap_rules` 是同一 axis 的 rule list，排序 key 是：

```python
(
    rule["z"],
    rule["rail_id"],
    rule["span_min"],
    rule["span_max"],
)
```

後續 `build_shared_rails()` 還會再把 x/y 兩個方向的 snap rules 合併，並改成依 z 分組。

### rail 內部格式

`_new_rail()` 建立的 internal rail 長這樣：

```python
{
    "axis": "x",
    "coord": 1.25,
    "min_coord": 1.0,
    "max_coord": 1.5,
    "members": [feature, ...],
    "lines_by_z": {
        0.0: [
            [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
            [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
        ],
    },
    "spans_by_z": {
        0.0: [
            (0.0, 5.0, 1.0),
            (6.0, 11.0, 1.5),
        ],
    },
}
```

`coord` 是 shared rail 的 checkerboard grid 座標，目前用 `min_coord` 與 `max_coord` 的中點：

```python
rail["coord"] = (rail["min_coord"] + rail["max_coord"]) / 2.0
```

### `_can_add_to_rail(rail, feature, merge_tol, eps, all_features)`

位置：`rail_builder.py`

責任：判斷一個 feature 能不能加入既有 rail。

它會檢查四類條件：

1. 合併後 rail 的座標範圍不能超過 `merge_tol`。
2. active z interval overlap 且 span overlap/touch 的 features，如果 target coord 不同，不能共用 rail。
3. active z interval overlap、不同 coord、端點距離太近但未相接時，不能共用 rail。
4. active z interval overlap 的中間 feature 如果座標落在新 rail 範圍內，而且 span 與候選 feature 或 rail member 重疊，會阻止合併。

這些規則是為了避免同一個 checkerboard node 在同一個 z event 被要求 snap 到兩個不同 target coordinates，或讓相鄰幾何在 corner 附近被不合理地合併。

### `_spans_overlap_or_touch(a_min, a_max, b_min, b_max, eps)`

位置：`rail_builder.py`

責任：判斷兩段一維 span 是否重疊或接觸。

判斷式：

```python
max(a_min, b_min) <= min(a_max, b_max) + eps
```

這表示只要兩段 overlap，或端點在 eps 容忍內碰到，就視為 overlap/touch。

### `_has_near_endpoint_conflict(rail, feature, merge_tol, eps)`

位置：`rail_builder.py`

責任：阻止 active z interval overlap 時，端點距離過近的不同座標線段合併。

概念例子：

```text
x=10, y span 0..10
x=11, y span 12..18
merge_tol = 2.2
```

兩條線的 y span gap 是 2，小於 `merge_tol`，代表它們在 corner 附近很接近。即使 span 沒有 overlap，也會被分到不同 rails，避免 corner 周圍的 checkerboard rail 合併過度。

### `_span_gap(a_min, a_max, b_min, b_max)`

位置：`rail_builder.py`

責任：計算兩段不重疊 span 之間的正距離。如果 overlap 或 touch，回傳 `0.0`。

### `_has_blocking_intermediate_feature(...)`

位置：`rail_builder.py`

責任：避免跨過 active z interval overlap 的中間衝突 feature 進行合併。

概念例子：

```text
x=10, y span 0..10
x=11, y span 5..30
x=12, y span 20..40
merge_tol = 2.2
```

如果嘗試把 `x=10` 與 `x=12` 放到同一 rail，`x=11` 位在兩者中間，而且 span 與兩側 feature 有重疊，因此會阻止合併。這讓 rail 合併保持幾何順序，不會穿越中間的 pattern feature。

### `_feature_overlaps_any(feature, others, eps)`

位置：`rail_builder.py`

責任：檢查某 feature 的 span 是否與一組 features 中任一個 span overlap/touch。它主要被 `_has_blocking_intermediate_feature()` 用來判斷中間 feature 是否真的是阻擋者。

### `_new_rail(axis, feature)`

位置：`rail_builder.py`

責任：用第一個 feature 建立一條新的 internal rail，然後立刻呼叫 `_add_to_rail()` 把 feature 加入。

### `_add_to_rail(rail, feature)`

位置：`rail_builder.py`

責任：把 feature 加入 rail，並同步更新 rail 的聚合欄位。

更新內容：

- `members.append(feature)`
- `min_coord`
- `max_coord`
- `coord`
- `lines_by_z[z].append(feature["line"])`
- `spans_by_z[z].append((span_min, span_max, coord))`

## 第四階段：輸出 public rail 與 snap rule

### `_serialize_rail(rail, rail_id)`

位置：`rail_builder.py`

責任：把 internal rail 轉成對外資料格式，移除 `members`、`spans_by_z` 這些內部判斷用資料。

輸出格式：

```python
{
    "axis": "x",
    "rail_id": 0,
    "coord": 1.25,
    "min_coord": 1.0,
    "max_coord": 1.5,
    "member_count": 2,
    "lines_by_z": {
        0.0: [
            [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
            [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
        ],
    },
}
```

欄位意義：

| 欄位 | 意義 |
| --- | --- |
| `axis` | rail 所屬方向，`"x"` 或 `"y"` |
| `rail_id` | 該 axis 內依 `coord` 排序後的 rail id |
| `coord` | checkerboard mesh 裡的 shared rail 座標 |
| `min_coord` / `max_coord` | 此 rail 代表的真實 pattern coordinate 範圍 |
| `member_count` | 合併了幾條 pattern feature lines |
| `lines_by_z` | 原始 pattern lines，依 snap z layer 分組 |

### `_rail_snap_rules(rail, rail_id)`

位置：`rail_builder.py`

責任：為 rail 中的每個 feature 建立 snap rule，並 deduplicate 完全相同的 rule。

輸出格式：

```python
{
    "axis": "x",
    "rail_id": 0,
    "rail_coord": 1.25,
    "target_coord": 1.0,
    "z": 0.0,
    "z_bottom": 0.0,
    "z_top": 20.0,
    "span_min": 0.0,
    "span_max": 5.0,
    "feature_id": 0,
}
```

欄位意義：

| 欄位 | 意義 |
| --- | --- |
| `axis` | 要 snap 的 rail 方向 |
| `rail_id` | 對應 shared rail |
| `rail_coord` | checkerboard mesh 原本的 rail 座標 |
| `target_coord` | 在此 z layer 要 snap 回去的真實 pattern 座標 |
| `z` | snap rule 生效的 z layer |
| `z_bottom` / `z_top` | 來源 feature 的 active z interval |
| `span_min` / `span_max` | 只 snap 這段 span 範圍內的 nodes/references |
| `feature_id` | 來源 feature id，方便 debug |

deduplicate key：

```python
(
    feature["z"],
    round(feature["span_min"], 8),
    round(feature["span_max"], 8),
    round(feature["coord"], 8),
)
```

因此 duplicate lines 不會產生重複 snap rule。

## `rail_builder.py` function quick reference

這一節用比較查表式的方式整理 `rail_builder.py` 裡每個 function 的輸入、輸出與副作用。

### `_round_key(value, decimals=4)`

用途：把座標轉成穩定的 dict key。

輸入：

```python
value = 0.00000001
decimals = 4
```

輸出：

```python
0.0
```

目前主要用在 `_line_to_feature()` 產生 `feature["z"]`，避免浮點誤差讓非常接近的 z 被分成不同 layer。

### `_line_to_feature(line, axis, feature_id, z_decimals=4)`

用途：把 raw line 轉成 feature dict。此 function 不判斷線段是否合法，合法性預期已由 `_classify_line()` 處理。

輸入：

```python
line = [[x1, y1], [x2, y2], [z_bottom, z_top]]
axis = "x" or "y"
feature_id = int
```

輸出：

```python
{
    "feature_id": int,
    "axis": "x" or "y",
    "coord": float,
    "z": float,
    "z_bottom": float,
    "z_top": float,
    "span_min": float,
    "span_max": float,
    "line": [[float, float], [float, float], [float, float]],
}
```

### `_spans_overlap_or_touch(a_min, a_max, b_min, b_max, eps)`

用途：檢查兩段 span 是否 overlap 或 touch。

輸出：

```python
True or False
```

在 shared rail 判斷裡，overlap 與 touch 都被視為有衝突風險，因為它們可能對同一段 checkerboard reference 產生不同 snap target。

### `_can_add_to_rail(rail, feature, merge_tol, eps, all_features)`

用途：合併判斷的核心 gatekeeper。

輸入：

```python
rail = internal_rail
feature = feature_dict
all_features = [feature_dict, ...]
```

輸出：

```python
True or False
```

內部判斷順序：

1. 檢查合併後 `new_max - new_min` 是否超過 `merge_tol + eps`。
2. 檢查 active z overlap 的既有 rail members 是否與新 feature span overlap/touch，且 target coord 不同。
3. 呼叫 `_has_near_endpoint_conflict()`。
4. 呼叫 `_has_blocking_intermediate_feature()`。

只要任一條件失敗就回傳 `False`。

### `_has_near_endpoint_conflict(rail, feature, merge_tol, eps)`

用途：阻止 active z overlap 時端點距離太近的不同座標線段合併。

輸出：

```python
True or False
```

判斷重點：

```python
eps < span_gap < merge_tol - eps
```

如果 span gap 落在這個範圍，代表兩段線沒有真的 overlap/touch，但距離小到可能屬於 corner 附近的鄰接幾何，因此不合併。

### `_span_gap(a_min, a_max, b_min, b_max)`

用途：回傳兩個 span 中間的空隙長度。

輸出：

```python
0.0       # overlap 或 touch
positive  # 兩段分離時的距離
```

### `_has_blocking_intermediate_feature(rail, feature, new_min, new_max, all_features, eps)`

用途：檢查 proposed rail coordinate range 中間是否有 active z overlap 的 feature 阻擋合併。

輸出：

```python
True or False
```

阻擋者必須同時滿足：

1. 不是目前 rail members，也不是這次要加入的 feature。
2. `other["coord"]` 位在 `new_min` 與 `new_max` 中間。
3. `other` 的 active z interval 與候選 feature 或 rail member overlap，且 span 也 overlap/touch。

這個 function 讓合併不會跨過中間的幾何 feature。

### `_feature_overlaps_any(feature, others, eps)`

用途：給 `_has_blocking_intermediate_feature()` 使用，判斷 feature 是否與任一候選 feature span overlap/touch。

輸出：

```python
True or False
```

### `_add_to_rail(rail, feature)`

用途：修改 internal rail，把 feature 加入 rail。

副作用：

```python
rail["members"].append(feature)
rail["min_coord"] = min(...)
rail["max_coord"] = max(...)
rail["coord"] = midpoint
rail["lines_by_z"][z].append(line)
rail["spans_by_z"][z].append((span_min, span_max, coord))
```

此 function 會直接 mutate 傳入的 `rail`。

### `_new_rail(axis, feature)`

用途：建立一條新的 internal rail。

輸出：

```python
{
    "axis": axis,
    "coord": feature["coord"],
    "min_coord": feature["coord"],
    "max_coord": feature["coord"],
    "members": [feature],
    "lines_by_z": {feature["z"]: [feature["line"]]},
    "spans_by_z": {
        feature["z"]: [
            (feature["span_min"], feature["span_max"], feature["coord"])
        ]
    },
}
```

實作上 `_new_rail()` 先建立空 rail，再呼叫 `_add_to_rail()`，所以初始化邏輯和後續加入 feature 的更新邏輯是同一套。

### `_build_axis_rails(features, axis, merge_tol, eps)`

用途：同 axis features 的主 grouping function。

輸出：

```python
public_rails = [public_rail, ...]
snap_rules = [snap_rule, ...]
```

如果 `features` 是空 list，直接回傳：

```python
([], [])
```

### `_serialize_rail(rail, rail_id)`

用途：把 internal rail 轉成 public rail。

輸出：

```python
{
    "axis": str,
    "rail_id": int,
    "coord": float,
    "min_coord": float,
    "max_coord": float,
    "member_count": int,
    "lines_by_z": {z: [line, ...]},
}
```

這個輸出會保存在 `rail_data["x_rails"]` 或 `rail_data["y_rails"]`。

### `_rail_snap_rules(rail, rail_id)`

用途：由一條 internal rail 的 members 產生 snap rules，並移除重複 rule。

輸出：

```python
[
    {
        "axis": str,
        "rail_id": int,
        "rail_coord": float,
        "target_coord": float,
        "z": float,
        "z_bottom": float,
        "z_top": float,
        "span_min": float,
        "span_max": float,
        "feature_id": int,
    },
    ...
]
```

### `build_shared_rails(lines, merge_tol, eps=0.01, z_decimals=4)`

用途：`rail_builder.py` 的公開 orchestration function。

輸出：

```python
{
    "x_rails": [public_rail, ...],
    "y_rails": [public_rail, ...],
    "x_list": [float, ...],
    "y_list": [float, ...],
    "snap_rules_by_z": {
        z: [snap_rule, ...],
    },
}
```

這是 `_get_feature_lines()` 直接消費的資料格式。

### `snap_rules_by_z`

`build_shared_rails()` 最後會把 x/y snap rules 合在一起，依 z 分組：

```python
{
    0.0: [
        {
            "axis": "x",
            "rail_id": 0,
            "rail_coord": 1.25,
            "target_coord": 1.0,
            "z": 0.0,
            "z_bottom": 0.0,
            "z_top": 0.0,
            "span_min": 0.0,
            "span_max": 5.0,
            "feature_id": 0,
        },
        {
            "axis": "y",
            "rail_id": 0,
            "rail_coord": 3.0,
            "target_coord": 3.0,
            "z": 0.0,
            "z_bottom": 0.0,
            "z_top": 0.0,
            "span_min": 0.0,
            "span_max": 5.0,
            "feature_id": 0,
        },
    ],
    10.0: [...],
}
```

排序規則：

- 先依 z 排序。
- 同一 z 內依 `(axis, rail_id, span_min)` 排序。

## 完整資料流範例

輸入：

```python
faces = [
    {"type": "LINE", "dim": [1, 0, 1, 5], "bottom_z": 0, "top_z": 0},
    {"type": "LINE", "dim": [1.5, 6, 1.5, 11], "bottom_z": 0, "top_z": 0},
    {"type": "LINE", "dim": [0, 3, 5, 3], "bottom_z": 0, "top_z": 0},
]
```

設定：

```python
merge_tol = 1.0
```

### 1. `_extract_lines()`

```python
lines = [
    [[1, 0], [1, 5], [0, 0]],
    [[1.5, 6], [1.5, 11], [0, 0]],
    [[0, 3], [5, 3], [0, 0]],
]
```

### 2. `_classify_line()`

```python
vertical_lines = [
    [[1, 0], [1, 5], [0, 0]],
    [[1.5, 6], [1.5, 11], [0, 0]],
]

horizontal_lines = [
    [[0, 3], [5, 3], [0, 0]],
]
```

### 3. `_line_to_feature()`

```python
vertical_features = [
    {
        "feature_id": 0,
        "axis": "x",
        "coord": 1.0,
        "z": 0.0,
        "z_bottom": 0.0,
        "z_top": 0.0,
        "span_min": 0.0,
        "span_max": 5.0,
        "line": [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
    },
    {
        "feature_id": 1,
        "axis": "x",
        "coord": 1.5,
        "z": 0.0,
        "z_bottom": 0.0,
        "z_top": 0.0,
        "span_min": 6.0,
        "span_max": 11.0,
        "line": [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
    },
]
```

因為兩條 x features 的 coord 差是 `0.5 <= merge_tol`，且同一個 snap z 的 span `0..5` 與 `6..11` 沒有 overlap/touch，也沒有 near endpoint conflict，所以可以共用 rail。

### 4. `_build_axis_rails()`

```python
x_rails = [
    {
        "axis": "x",
        "rail_id": 0,
        "coord": 1.25,
        "min_coord": 1.0,
        "max_coord": 1.5,
        "member_count": 2,
        "lines_by_z": {
            0.0: [
                [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
                [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
            ],
        },
    }
]

y_rails = [
    {
        "axis": "y",
        "rail_id": 0,
        "coord": 3.0,
        "min_coord": 3.0,
        "max_coord": 3.0,
        "member_count": 1,
        "lines_by_z": {
            0.0: [
                [[0.0, 3.0], [5.0, 3.0], [0.0, 0.0]],
            ],
        },
    }
]
```

### 5. `_get_feature_lines(..., return_details=True)`

最後回傳：

```python
group_lines_v = [
    {
        0.0: [
            [[1.0, 0.0], [1.0, 5.0], [0.0, 0.0]],
            [[1.5, 6.0], [1.5, 11.0], [0.0, 0.0]],
        ]
    }
]

group_lines_h = [
    {
        0.0: [
            [[0.0, 3.0], [5.0, 3.0], [0.0, 0.0]],
        ]
    }
]

x_list = [1.25]
y_list = [3.0]

snap_rules_by_z = {
    0.0: [
        {
            "axis": "x",
            "rail_id": 0,
            "rail_coord": 1.25,
            "target_coord": 1.0,
            "z": 0.0,
            "z_bottom": 0.0,
            "z_top": 0.0,
            "span_min": 0.0,
            "span_max": 5.0,
            "feature_id": 0,
        },
        {
            "axis": "x",
            "rail_id": 0,
            "rail_coord": 1.25,
            "target_coord": 1.5,
            "z": 0.0,
            "z_bottom": 0.0,
            "z_top": 0.0,
            "span_min": 6.0,
            "span_max": 11.0,
            "feature_id": 1,
        },
        {
            "axis": "y",
            "rail_id": 0,
            "rail_coord": 3.0,
            "target_coord": 3.0,
            "z": 0.0,
            "z_bottom": 0.0,
            "z_top": 0.0,
            "span_min": 0.0,
            "span_max": 5.0,
            "feature_id": 0,
        },
    ]
}
```

## 合併規則整理

可以合併到同一 shared rail 的常見情況：

- active z interval 不 overlap 的線段可以合併，即使它們在 xy span 上 overlap，因為它們不會在同一段 extrusion interval 同時有效。
- active z interval overlap 的線段如果 span 不 overlap/touch，且沒有 near endpoint conflict，可以合併。
- 完全重複的線段可以進同一 rail，但只會產生一條 snap rule。

不能合併的常見情況：

- 合併後 `max_coord - min_coord > merge_tol + eps`。
- active z interval overlap、span overlap/touch、但 target coord 不同。
- active z interval overlap、不同 coord、span endpoint gap 介於 `eps` 與 `merge_tol - eps` 之間。
- active z interval overlap 的中間 feature 位於 proposed rail 的座標範圍內，且 span 與候選 feature 或 rail member overlap/touch。
- 線段不是水平或垂直。
- z interval 不合法，例如 `top_z < bottom_z`。

## 後續如何被 mesher 使用

`OptimalMesh25D.set_pattern()` 會把 `_get_feature_lines(..., return_details=True)` 的結果存到物件狀態：

```python
self.group_lines_v = group_lines_v
self.group_lines_h = group_lines_h
self.x_list = x_list
self.y_list = y_list
self.snap_rules_by_z = snap_rules_by_z
self.rails = {"x": x_rails, "y": y_rails}
```

後續：

- `x_list` / `y_list` 會成為 checkerboard mesh 必須包含的 grid line。
- `rails` 會用來建立 rail node index，記錄每條 shared rail 上有哪些結構節點，以及這些節點在 span axis 上的排序。
- `node_axis_rail_ids` 會記錄每個節點分別屬於哪條 x rail 與 y rail，讓 snap 時可以找到 shared-rail 交點。
- `snap_rules_by_z` 會在 `apply_snap_rules_at_z(z)` 時查出該 z layer 要執行的座標修正。

### `apply_snap_rules_at_z(z)` 的套用模型

snap rule 的 `span_min` / `span_max` 是用「結構上的 shared rail 座標」查節點，不是用節點目前已被修改後的座標。這點很重要，因為同一個 z layer 可能同時有 x rule 與 y rule 要作用在同一個 corner。如果逐條 rule 立即改座標，某些角點會因為另一軸尚未 snap 而漏選。

目前的套用流程是：

1. 取出同一個 z layer 的全部 snap rules。
2. 用 `_rules_by_axis_and_rail()` 依 axis 與 rail id 分組，供 corner lookup 使用。
3. 每條 rule 先用 `_span_node_ids()` 找出自己 rail 上、structural span 落在 `[span_min, span_max]` 的節點。
4. 再用 `_coupled_corner_node_ids()` 補上同 z layer 的 perpendicular snap corner：如果 x rule 的 target x 落在某條 y rule 的 x span 內，而且 y rule 的 target y 也落在該 x rule 的 y span 內，兩條 shared rail 的交點就應該同時被這兩條 rule 納入。
5. 所有 rule 都只先寫入 `target_values` / `target_masks`。
6. 最後一次把 x 與 y 兩個座標軸的 target values 寫回 nodes。

因此同一個 z layer 的 snap 是「先收集、後寫回」的 atomic update。這避免 x/y rules 因套用順序互相影響，也讓 shared rail corner 可以正確移回真實幾何角點。

### corner coupling 範例

例如兩個 box：

```python
faces = [
    {"type": "BOX", "dim": [0, 0, 10, 10], "bottom_z": 0, "top_z": 10},
    {"type": "BOX", "dim": [2, 1, 8, 5], "bottom_z": 11, "top_z": 20},
]
```

在 `element_size=11.0`、`ratio=0.2` 時，face1 與 face2 的相近邊會共用 shared rails：

```python
x_list = [1.0, 9.0]
y_list = [0.5, 5.0, 10.0]
```

z=11 時，face2 的底邊真實位置是 `y=1`、span `x=2..8`；左右邊真實位置是 `x=2` / `x=8`、span `y=1..5`。結構節點一開始位在 shared rail 交點 `(1, 0.5)` 與 `(9, 0.5)`，單看任何一條 rule 的 structural span 都不足以選到正確 corner。

corner coupling 會看到：

- x rule `target_coord=2` 的 y span `1..5` 包含 y rule 的 `target_coord=1`。
- y rule `target_coord=1` 的 x span `2..8` 包含 x rule 的 `target_coord=2`。
- 所以 shared rail 交點 `(1, 0.5)` 要同時套用 x=2 與 y=1，最後變成 `(2, 1)`。
- 右下角同理會從 `(9, 0.5)` 變成 `(8, 1)`。

因此整個演算法的精神是：前處理時盡量把相近 pattern lines 合併成較少的 checkerboard rails，降低 mesh 複雜度；真正 drag 到特定 z layer 時，再用同 z 的 snap rules 共同決定節點應該移到哪裡，最後一次寫回真實幾何座標。
