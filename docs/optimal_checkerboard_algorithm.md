# optimal_checkerboard 演算法與資料流

這份文件說明 `optimal_checkerboard` 如何將分層 2.5D 幾何轉成：

- shared X/Y rails；
- 一份可重複使用的 2D quadrilateral mesh；
- 依 Z 分組的 snap/restore rules；
- 沿 Z 拉伸的 3D hexahedral mesh。

對外資料格式請先參考 [輸入與輸出資料格式](data_formats.md)。本文件聚焦在演算法責任、設計理由與重要 invariant。

## 1. 核心概念

### 1.1 2.5D 模型

本專案處理的幾何具有以下特徵：

- 複雜度主要位於 XY 平面。
- 幾何與材料沿 Z 方向分層。
- 同一個 XY topology 可以在多個 Z interval 重複使用。

因此不需要一次建立並管理完整 3D mesh。演算法先建立一份 2D mesh，再依 Z event 移動部分節點並進行 extrusion。

### 1.2 Pattern line

pattern line 是從 face boundary 抽出的水平或垂直線段：

```python
[
    [x1, y1],
    [x2, y2],
    [z_bottom, z_top],
]
```

`[z_bottom, z_top]` 表示這條 projected edge 的 active Z interval。

- vertical line：X 固定、Y 改變，形成 X rail candidate。
- horizontal line：Y 固定、X 改變，形成 Y rail candidate。

### 1.3 Shared rail

shared rail 是 checkerboard mesh 中的一條結構 grid line。多條相近的 pattern lines 若不會在相同 XY span 和 active Z interval 對同一批節點要求不同 target coordinates，就可以共用 rail。

例如：

```text
feature A: x=1.0, y span=0..5,  z=0..10
feature B: x=1.5, y span=6..11, z=0..10
```

若合併規則允許，兩者可以形成：

```text
shared rail x=1.25
```

rail coordinate 目前使用 members 最小與最大座標的中點：

```python
rail_coord = (min_coord + max_coord) / 2
```

### 1.4 Snap rule

2D mesh 預設使用 shared rail coordinate。到達 feature 的 `bottom_z` 時，snap rule 將指定 span 上的 rail nodes 移到真實 pattern coordinate。

### 1.5 Restore rule

feature 到達 `top_z` 時，不會立即把 nodes 還原。restore rule 先進入 buffer，在下一個更高的 Z event 才套用。

因此：

```text
bottom_z layer     feature 開始出現
top_z layer        feature 仍然存在
next higher layer  還原到 shared rail
```

## 2. 整體資料流

```text
Obj hierarchy
    │
    ├─ set_position_abs()
    ├─ mesh_domain_from_obj()
    └─ _pattern_faces_from_obj()
           │
           ▼
raw face dictionaries
           │
           └─ _extract_lines()
                  │
                  ▼
canonical horizontal/vertical lines
                  │
                  ├─ _classify_line()
                  └─ _line_to_feature()
                         │
                         ▼
normalized X/Y features
                         │
                         └─ build_shared_rails()
                                │
                                ├─ shared rails
                                ├─ snap_rules_by_z
                                └─ restore_rules_by_z
                                       │
                                       ▼
generate_checkerboard_mesh()
                                       │
                                       ├─ Mesh2D.nodes
                                       ├─ Mesh2D.elements
                                       └─ rail_node_index
                                              │
                                              ▼
apply_snap_rules_at_z()
                                              │
                                              ▼
Dragger._organize() + Dragger._drag()
                                              │
                                              ▼
3D nodes + hexahedra + component ids
```

主要 orchestration 位於：

- `src/optimal_checkerboard/mesher.py`
- `src/optimal_checkerboard/algorithms/feature_lines.py`
- `src/optimal_checkerboard/algorithms/rail_builder.py`
- `src/optimal_checkerboard/algorithms/drag.py`

## 3. Stage A：`Obj` 轉成 pattern faces

公開入口：

```python
OptimalMesh25D.set_pattern_obj(obj, element_size, ratio=0.1)
```

流程：

```python
obj.set_position_abs(0, 0, 0)
mesh_domain = mesh_domain_from_obj(obj)
faces = self._pattern_faces_from_obj(obj)
self._set_pattern(faces, element_size, ratio, mesh_domain)
```

### 3.1 Absolute coordinates

`Obj.set_position_abs()` 由 root 往下傳遞 origin：

- root 的 local coordinates 轉成 absolute coordinates；
- metal begin/end 轉成 absolute Z；
- mesh line/face 轉成 absolute XY；
- child object 相對於 parent origin 放置。

### 3.2 Pattern face 來源

`_pattern_faces_from_obj()` 收集：

1. `Obj.face`。
2. 每個 `Metal.ranges`。
3. 每個 `Metal.holes`。
4. `Mesh.line`。
5. `Mesh.face`。
6. 所有 child objects 的上述幾何。

Metal 沒有 ranges 或 holes 時不增加新 pattern face，因為它沿用 parent object footprint。

### 3.3 Mesh domain 與 pattern faces 的差異

`mesh_domain` 決定整份 2D mesh 的外部 footprint。

`pattern faces` 則決定內部必要 rails 與 snap events。

root face 同時可能扮演這兩個角色，但兩者資料用途不同。

`Obj.layers` 在這個階段用來累積 `Obj.thk`，進而決定 object face 的 active Z interval；layer material 不會自動轉成 `build()` 使用的 area/material stack。

### 3.4 CYLINDER 接口

CYLINDER root 可保留為 mesh domain，並由 domain dispatcher 路由到：

```python
checkerboard_mesh_cylinder(
    domain,
    x_list,
    y_list,
    element_size,
)
```

這個 repository 刻意保留 dispatcher contract；實際 cylinder mesh generator 可由外部專案提供。

CYLINDER pattern face 不會進入 shared-rail pipeline，因為該演算法處理的是 orthogonal horizontal/vertical edges。

## 4. Stage B：Face boundary extraction

入口：

```python
_extract_lines(faces)
```

輸入：

```python
{
    "type": "BOX" | "LINE" | "POLYGON",
    "dim": ...,
    "bottom_z": z0,
    "top_z": z1,
}
```

輸出：

```python
lines = [
    [[x1, y1], [x2, y2], [z_bottom, z_top]],
    ...
]
```

### 4.1 BOX

```python
{
    "type": "BOX",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

轉成下、右、上、左四條邊：

```python
[
    [[x1, y1], [x2, y1], [z0, z1]],
    [[x2, y1], [x2, y2], [z0, z1]],
    [[x2, y2], [x1, y2], [z0, z1]],
    [[x1, y2], [x1, y1], [z0, z1]],
]
```

### 4.2 LINE

```python
{
    "type": "LINE",
    "dim": [x1, y1, x2, y2],
    "bottom_z": z0,
    "top_z": z1,
}
```

轉成：

```python
[[x1, y1], [x2, y2], [z0, z1]]
```

### 4.3 POLYGON

每個 loop 先經過 `normalize_polygon_loops()`：

1. 移除重複 closure point。
2. 檢查至少三個點。
3. 檢查面積非零。
4. 依 signed area 判定 hull/hole。
5. 將每一對相鄰點轉成 line，包含最後一點到第一點。

### 4.4 Boundary extraction invariant

- 每個 face 必須有 `bottom_z` 與 `top_z`。
- Z interval 必須排序正確。
- `_extract_lines()` 統一資料格式，但不負責所有方向判斷。
- diagonal 與 zero-length 檢查在下一階段進行。

## 5. Stage C：Line classification 與 feature normalization

### 5.1 `_classify_line()`

```python
vertical_lines, horizontal_lines = _classify_line(lines, eps=0.01)
```

判斷：

```text
vertical:
    abs(x1 - x2) <= eps
    abs(y1 - y2) > eps

horizontal:
    abs(y1 - y2) <= eps
    abs(x1 - x2) > eps
```

以下會被拒絕：

- diagonal line；
- zero-length line；
- 反向 Z interval；
- legacy line 兩端 Z 不一致。

### 5.2 `_line_to_feature()`

vertical line 轉成 X feature：

```python
{
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

欄位意義：

| 欄位 | 意義 |
| --- | --- |
| `feature_id` | 該 axis pipeline 內的來源 line id |
| `axis` | `"x"` 或 `"y"` |
| `coord` | line 的固定座標 |
| `z` | snap event key，等於 rounded `z_bottom` |
| `z_bottom` | feature 開始高度 |
| `z_top` | feature 結束高度 |
| `span_min` | 另一個 XY 軸的最小值 |
| `span_max` | 另一個 XY 軸的最大值 |
| `line` | float-normalized canonical line |

X 與 Y features 分開編號，因此不同 axis 可以有相同 `feature_id`。

## 6. Stage D：Shared rail grouping

主要入口：

```python
build_shared_rails(
    lines,
    merge_tol,
    eps=0.01,
    z_decimals=4,
)
```

回傳：

```python
{
    "x_rails": [...],
    "y_rails": [...],
    "x_list": [...],
    "y_list": [...],
    "snap_rules_by_z": {...},
    "restore_rules_by_z": {...},
}
```

### 6.1 X/Y pipelines

```python
x_rails = _build_axis_rails(vertical_features, "x", merge_tol, eps)
y_rails = _build_axis_rails(horizontal_features, "y", merge_tol, eps)
```

兩個方向使用同一套一維 grouping 邏輯：

- X rail 比較 X coordinate，span 是 Y。
- Y rail 比較 Y coordinate，span 是 X。

### 6.2 Greedy grouping

features 先依下列欄位排序：

```python
(
    coord,
    z_bottom,
    z_top,
    span_min,
)
```

對每個 feature：

1. 排除座標距離已超過 merge tolerance 的舊 rails。
2. 找出所有可加入的 candidate rails。
3. 模擬加入後是否保持相鄰 rail 順序。
4. 在合法 candidates 中選擇距離目前 rail coordinate 最近者。
5. 沒有合法 candidate 時建立新 rail。

這是具安全條件的 greedy grouping，不保證求得 rail 數量的全域數學最小值。

### 6.3 Sliding active window

features 依 coordinate 遞增處理。若：

```python
feature["coord"] - rail["max_coord"] > merge_tol + eps
```

則目前與後續 features 都不可能再加入該 rail。演算法以 `active_start` 排除這些 rails，減少不必要比較。

## 7. Rail merge safety rules

一個 feature 要加入既有 rail，必須同時通過 rail 內衝突與相鄰 rail topology 檢查。

### 7.1 Coordinate range

合併後：

```python
new_max_coord - new_min_coord <= merge_tol + eps
```

注意這是整條 rail members 的總座標範圍，不只是 feature 到目前 midpoint 的距離。

### 7.2 Active Z overlap

兩個 features 的 active Z interval overlap/touch 判斷概念為：

```python
max(a.z_bottom, b.z_bottom) <= min(a.z_top, b.z_top) + eps
```

不同 Z interval 的 features 有較高機會共用 rail，因為它們不會在同一 extrusion interval 同時要求不同 target coordinates。

### 7.3 XY span overlap/touch

```python
max(a.span_min, b.span_min) <= min(a.span_max, b.span_max) + eps
```

active Z overlap 且 XY span overlap/touch 的不同 coordinates 不能共用 rail，否則同一 node 可能收到互相衝突的 snap targets。

### 7.4 Near endpoint conflict

即使 spans 沒有接觸，若兩條 active-Z-overlapping features 的端點 gap 很小，合併仍可能破壞 corner 附近的 topology。

概念條件：

```python
eps < span_gap < merge_tol - eps
```

這類 features 會保持在不同 rails。

### 7.5 Blocking intermediate feature

假設：

```text
x=10 feature
x=11 feature
x=12 feature
```

若嘗試直接合併 `x=10` 與 `x=12`，且 `x=11` 在 active Z 與 span 上構成衝突，它會成為 blocking intermediate feature。

這避免 rail grouping 跨過中間幾何。

### 7.6 Adjacent rail order

rail 內合法不代表整體 topology 一定合法。加入 feature 後還要模擬所有 rails，確認相鄰 rails 在以下狀態都不會反序：

- 兩條 rail 都在 default coordinate；
- 左／下 rail snap，右／上 rail 保持 default；
- 右／上 rail snap，左／下 rail 保持 default；
- active Z overlap 時兩條 rail 同時 snap。

對相鄰 `left` 與 `right` rails，主要 invariant 是：

```text
left coordinate < right coordinate
```

這個順序必須同時適用於 default rail coordinates 與可能同時出現的 feature target coordinates。

否則 quadrilateral 可能交錯，3D extrusion 後可能形成負面積或負體積元素。

## 8. Rail 與 rule serialization

### 8.1 Internal rail

grouping 階段使用：

```python
{
    "axis": "x",
    "coord": 1.25,
    "min_coord": 1.0,
    "max_coord": 1.5,
    "members": [feature, ...],
    "lines_by_z": {
        0.0: [line, ...],
    },
    "spans_by_z": {
        0.0: [
            (span_min, span_max, target_coord),
        ],
    },
}
```

`members` 和 `spans_by_z` 只供 grouping 判斷使用。

### 8.2 Public rail

`_serialize_rail()` 移除內部欄位：

```python
{
    "axis": "x",
    "rail_id": 0,
    "coord": 1.25,
    "min_coord": 1.0,
    "max_coord": 1.5,
    "member_count": 2,
    "lines_by_z": {
        0.0: [line, ...],
    },
}
```

rails 依 `coord` 排序後取得 axis-local `rail_id`。

### 8.3 Snap rule

每個 deduplicated feature 產生：

```python
{
    "kind": "snap",
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

### 8.4 Restore rule

restore rule 使用相同結構：

```python
{
    "kind": "restore",
    "axis": "x",
    "rail_id": 0,
    "rail_coord": 1.25,
    "target_coord": 1.25,
    "z": 20.0,
    "z_bottom": 0.0,
    "z_top": 20.0,
    "span_min": 0.0,
    "span_max": 5.0,
    "feature_id": 0,
}
```

rules 最後依 Z 分組：

```python
snap_rules_by_z = {
    bottom_z: [snap_rule, ...],
}

restore_rules_by_z = {
    top_z: [restore_rule, ...],
}
```

## 9. Stage E：2D checkerboard mesh

公開入口：

```python
mesh2d = mesher.mesh_checkerboard()
```

domain dispatcher：

```python
generate_checkerboard_mesh(
    domain,
    x_list,
    y_list,
    element_size,
)
```

### 9.1 BOX domain

BOX generator 先將 root bounds 加入 rail lists：

```python
mesh_x_list = unique_sorted(x_list + [xmin, xmax])
mesh_y_list = unique_sorted(y_list + [ymin, ymax])
```

並檢查所有 shared rails 都在 root bounds 內。

### 9.2 Axis densification

對相鄰必要 coordinates：

```python
element_count = max(
    1,
    ceil(interval_length / element_size),
)
```

再以 `linspace` 建立 interval nodes。這保證：

- 每條 required rail 一定成為 grid line；
- 每段至少有一個 element；
- 實際 element size 不超過偏好值，浮點誤差除外。

### 9.3 Mesh arrays

```text
nodes     (n_nodes, 3), float64
elements  (n_elements, 4), int32
```

nodes 以 X varying fastest 的 row-major structured grid 排列。

element connectivity：

```python
[
    bottom_left,
    bottom_right,
    top_right,
    top_left,
]
```

### 9.4 Custom mesh

```python
mesher.mesh_assignment(mesh2d)
```

允許外部 mesher 提供 quadrilateral mesh。shared rail coordinates 必須存在於 mesh nodes 中，而且 snapping 期間 topology 與 node ids 必須保持穩定。

## 10. Rail node index

建立 2D mesh 後，`_build_rail_node_index()` 為每條 rail 建立：

```python
{
    "node_ids": sorted_node_ids,
    "span_values": sorted_span_coordinates,
}
```

X rail：

- coordinate axis：X。
- span axis：Y。

Y rail：

- coordinate axis：Y。
- span axis：X。

只索引 `elements` 實際引用的 active nodes。

對單一 snap rule，可使用 binary search 找到 span：

```python
lo = searchsorted(span_values, span_min - eps)
hi = searchsorted(span_values, span_max + eps)
node_ids = sorted_node_ids[lo:hi]
```

這避免每個 rule 掃描全部 mesh nodes。

## 11. Stage F：套用 snap/restore rules

入口：

```python
mesher.apply_snap_rules_at_z(z)
```

流程：

1. 若目前 Z 高於 restore buffer 的來源 Z，取出 buffered restore rules。
2. 取得目前 Z 的 snap rules。
3. 將 restore 與 snap rules 組成同一批 atomic updates。
4. 計算每條 rule 影響的 span nodes。
5. 加入需要一起移動的 coupled corner nodes。
6. 寫入所有 target coordinates。
7. 將目前 Z 的 top restore rules 放入 buffer。

### 11.1 為什麼 batch apply

同一 Z 可能同時發生：

- 前一個 feature 的 restore；
- 新 feature 的 snap；
- X/Y perpendicular snaps。

程式先計算所有 node selections，再統一更新 coordinates，避免 rule 執行順序改變後續 node lookup 結果。

### 11.2 Snap 優先於同 node restore

當 staged restore 與 current snap 影響同一個 coordinate axis/node 時，current snap 的 update 排在後面，因此目前 layer 的新 feature target 會成為最終座標。

### 11.3 Coupled corners

X rule 與 Y rule 若 target spans 相交，交點 node 可能需要同時更新 X 與 Y。

判斷概念：

```python
x_rule.span contains y_rule.target_coord
and
y_rule.span contains x_rule.target_coord
```

符合時，corner node 會加入兩條 rules 的 affected node set。

### 11.4 Top-Z delayed restore

假設 feature：

```text
z_bottom = 0
z_top    = 10
```

事件：

```text
apply z=0   -> snap
apply z=10  -> feature 仍維持，restore 只 staged
apply z>10  -> consume restore，回到 rail default
```

Z traversal 應由低到高。若要重新開始一輪 traversal，先呼叫：

```python
mesher.reset_snap_state()
```

## 12. Stage G：3D organize 與 drag

入口：

```python
dragger = mesher.build(obj_list)
```

每個 Z interval：

1. `apply_snap_rules_at_z(z_begin)`。
2. 若已有上一層 3D top nodes，同步被 snap 的 XY。
3. 重新計算 2D element areas。
4. `_organize(areas)` 分配 material/component ids。
5. `_drag(element_size, z_begin, z_end)` 建立 3D nodes 與 hexahedra。

`build()` 只在 `obj_list` 中每個 layer 的 `z_begin` 執行 snap/restore。因此 `obj_list` 必須包含所有會改變 pattern geometry 的 Z events。`Obj` hierarchy 不會自動轉換成這份 3D material stack。

### 12.1 Area selection

每個 2D quadrilateral 以四個 corners 進行幾何判斷。四點都在 area 內時才選入。

- BOX：比較 element min/max bounds。
- CYLINDER：四點都需位於半徑內。
- POLYGON：四點都需位於任一 hull 且不在 holes。

polygon boundary 使用 inclusive test，位於邊界上的 corners 視為 inside。

### 12.2 Material assignment

`Dragger.comps`：

```python
{"EMPTY": 0}
```

新材料在第一次出現時取得新的整數 id。

Area metals 的處理順序：

1. 計算 `NORMAL` metal 的候選 volume。
2. 處理 `CONTINUE`。
3. 處理 `CONVERT`。
4. 處理 `NORMAL` density assignment。
5. 剩餘 elements 指派 area material。

### 12.3 Z subdivision

對距離：

```python
distance = end - begin
```

drag subdivision：

```python
drag_num = max(1, floor(distance / preferred_element_size))
actual_size = distance / drag_num
```

此處使用的是 Z direction element size。

### 12.4 Hexahedral connectivity

每個 selected quadrilateral 在相鄰 Z planes 之間形成：

```python
[
    bottom_0,
    bottom_1,
    bottom_2,
    bottom_3,
    top_0,
    top_1,
    top_2,
    top_3,
]
```

## 13. 重要 invariant

修改演算法時應維持：

1. Pattern edges 必須是水平或垂直。
2. 每個 feature 都有合法 active Z interval。
3. 同 rail 的 active-Z-overlapping features 不能對同一 span 產生不同 targets。
4. Adjacent rails 在 default 與 snap 狀態下都不能反序。
5. 2D snapping 只改 coordinates，不改 element topology。
6. `rail_node_index` 中的 node ids 在整輪 traversal 保持有效。
7. Restore 在 `top_z` 下一個更高 Z 才執行。
8. X/Y 同 Z corner snaps 必須同步處理。
9. `Dragger` 預先配置 arrays，只能以 `node_num`／`element_num` 判斷有效範圍。

## 14. 複雜度概觀

令：

- `F`：feature line 數量；
- `R`：shared rail 數量；
- `N`：2D node 數量；
- `Q_z`：某一 Z event 的 rules 數量。

大致成本：

```text
feature sorting          O(F log F)
rail grouping            依 active candidate rails 數量而定
rail node index          O(N log N) 級別的排序工作
single Z snap            O(Q_z log nodes_per_rail + touched_nodes)
2D mesh generation       O(N + number_of_quads)
3D drag                  O(generated_nodes + generated_hexes)
```

實際效能收益主要來自：

- shared rails 減少必要 XY grid lines；
- active rail window 減少 grouping candidates；
- sorted rail node index 加速 span queries；
- NumPy vectorization 批次建立 3D nodes/elements。

## 15. Extension points

### 15.1 外部 2D mesher

流程：

```python
mesher.set_pattern_obj(...)
snap_faces = mesher.get_snap_faces()
custom_mesh = external_mesher(snap_faces)
mesher.mesh_assignment(custom_mesh)
```

### 15.2 外部 CYLINDER mesher

保留的函式 contract：

```python
checkerboard_mesh_cylinder(
    domain,
    x_list,
    y_list,
    element_size,
)
```

預期輸出需與 BOX generator 相容，並可被 dispatcher 整合：

```python
nodes
elements
mesh_x_list
mesh_y_list
```

若外部實作直接提供完整 `Mesh2D`，也可以繞過內建 generator，使用 `mesh_assignment()`。

### 15.3 新 pattern primitive

若新增可轉換為 orthogonal edges 的 primitive，主要擴充位置是：

1. `OptimalMesh25D._append_pattern_face()` 的 type validation。
2. `_extract_lines()` 的 boundary conversion。
3. 必要的 input validation 與 regression tests。

shared-rail grouping 本身只依賴 canonical lines，不需理解原始 face type。

## 16. 測試對照

| 行為 | 測試 |
| --- | --- |
| BOX 2D mesh node/element ordering | `tests/test_checkerboard_mesh_box.py` |
| root domain 與 mesh dispatcher | `tests/test_mesh_checkerboard_box.py` |
| line classification | `tests/test_classify_line.py` |
| face boundary extraction | `tests/test_extract_lines.py` |
| shared rail grouping | `tests/test_build_shared_rails.py` |
| feature-line orchestration | `tests/test_get_feature_lines.py` |
| pattern conversion | `tests/test_set_pattern.py` |
| custom mesh assignment | `tests/test_mesh_assignment.py` |
| snap/restore execution | `tests/test_apply_snap_rules_at_z.py` |
| 2D area/material drag engine | `tests/test_drag_engine.py` |
| full `OptimalMesh25D.build()` | `tests/test_optimal_mesh_build.py` |

修改 rail compatibility 時，至少應覆蓋：

- same-Z span overlap；
- same-Z span separation；
- cross-Z sharing；
- near endpoint conflict；
- blocking intermediate feature；
- adjacent rail order；
- delayed restore；
- same-Z X/Y corner coupling。

## 17. 完整小型範例

輸入：

```python
faces = [
    {
        "type": "LINE",
        "dim": [1, 0, 1, 5],
        "bottom_z": 0,
        "top_z": 10,
    },
    {
        "type": "LINE",
        "dim": [1.5, 6, 1.5, 11],
        "bottom_z": 0,
        "top_z": 10,
    },
    {
        "type": "LINE",
        "dim": [0, 3, 5, 3],
        "bottom_z": 0,
        "top_z": 10,
    },
]
```

假設：

```python
merge_tol = 1.0
```

抽線後：

```python
[
    [[1.0, 0.0], [1.0, 5.0], [0.0, 10.0]],
    [[1.5, 6.0], [1.5, 11.0], [0.0, 10.0]],
    [[0.0, 3.0], [5.0, 3.0], [0.0, 10.0]],
]
```

分類：

```text
vertical features   x=1.0, x=1.5
horizontal feature  y=3.0
```

由於兩條 vertical features 的 spans 分離且沒有 endpoint conflict，可以形成：

```python
x_list = [1.25]
y_list = [3.0]
```

Z=0 snap rules：

```text
x rail 0, y span 0..5   -> x=1.0
x rail 0, y span 6..11  -> x=1.5
y rail 0, x span 0..5   -> y=3.0
```

Z=10 呼叫時，restore rules 先 staged；下一個更高 Z event 才將兩段 X rail 還原到 `x=1.25`。

這個例子展示此專案的核心：以較少的 structural rails 表示多個分層 pattern coordinates，再以 Z-aware rules 恢復真實幾何。
