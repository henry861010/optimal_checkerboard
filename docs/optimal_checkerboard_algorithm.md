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

### 1.5 Active state 與 restore metadata

snap rule 的 active interval 為兩端包含：

```text
z_bottom <= z <= z_top
```

`apply_snap_rules_at_z(z)` 每次都從 structural baseline 重建該 Z 的完整
active state。它先恢復上一個 managed state 改動過的 rail nodes，再原子性
套用目前所有 active rules。因此 `top_z` 平面仍有 feature；任一嚴格大於
`top_z` 的查詢中，它不再 active，對應 nodes 就回到 baseline。這不依賴
遍歷順序，restore 不是另一個執行佇列。

`restore_rules_by_z` 仍保留為以 feature `top_z` 索引的生命週期 metadata，
方便 caller 列舉完整 Z boundaries；它不是執行佇列。

## 2. 整體資料流

```text
Obj hierarchy
  ├─ set_position_abs() / mesh_domain_from_obj()
  └─ _pattern_faces_from_obj()
       │
       ▼
raw faces -> _extract_lines() -> _classify_line() -> normalized X/Y features
       │
       ▼
build_shared_rails()
  ├─ shared rails + snap_rules_by_z
  └─ restore_rules_by_z (lifecycle metadata)
       │
       ▼
snap-plan preflight on structural axes
  └─ unsafe optimized plan -> rebuild exact-coordinate rails
       │
       ▼
derive/densify full structured axes (including filler stations)
       │
       ▼
snap-plan preflight on full axes
  └─ unsafe optimized plan -> rebuild exact-coordinate rails and retry
       │
       ▼
generate Mesh2D + verified STRUCTURED_BOX metadata/direct rail index
       │
       ▼
apply complete active snap state at z
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

### 3.4 Non-BOX footprint 的 fail-closed boundary

CYLINDER root 仍會被 domain dispatcher 路由到保留的 symbol：

```python
checkerboard_mesh_cylinder(
    domain,
    x_list,
    y_list,
    element_size,
)
```

但這不是可用的 high-level extension contract。CYLINDER pattern face 不會進入
shared-rail pipeline，且目前無法證明 POLYGON/CYLINDER footprint 的完整
2D partition。因此：

- `mesh_checkerboard()` 對這些 footprint 尚未實作；
- 帶 explicit POLYGON/CYLINDER domain 的 `mesh_assignment()` 會拒絕；
- `build()` 只接受 explicit BOX domain。

外部 cylinder `Mesh2D` 不能繞過這個 proof obligation。要開放該路徑，必須先
實作對應 domain partition 與 build-boundary validator。

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
vertical_lines, horizontal_lines = _classify_line(lines)
```

判斷：

```text
vertical:
    abs(x1 - x2) <= scale_aware_noise
    abs(y1 - y2) > scale_aware_noise

horizontal:
    abs(y1 - y2) <= scale_aware_noise
    abs(x1 - x2) > scale_aware_noise
```

`scale_aware_noise` 最多為座標尺度的數個 ULP，只用來容忍浮點表示雜訊。
即使 caller 傳入更大 `eps`，也只會當作這個上限的 cap，不會把真正的斜線
重新解釋為 axis-aligned edge。這個 noise allowance 不參與 topology equality；兩個
rail/target coordinates 即使只相差 1 ULP 也不會 alias。

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
| `z` | snap event key，精確等於 `z_bottom` |
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
    eps=None,
    z_decimals=None,
)
```

`z_decimals` 只為來源相容而保留，不會對 Z event rounding。`eps` 只能限制
line-classification 的 scale-aware floating-point noise；rail targets、spans、merge bounds
與 Z events 均使用 exact topology values。

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
3. 以目前左右／上下鄰居與最近 pinned coordinates，檢查加入後是否保持
   相鄰 rail 順序。
4. 在合法 candidates 中選擇距離目前 rail coordinate 最近者。
5. 沒有合法 candidate 時建立新 rail。

這是具安全條件的 greedy grouping，不保證求得 rail 數量的全域數學最小值。

在 greedy loop 前，所有 feature pairs 的 XY span overlap、active-Z conflict 與
incompatibility 會預先編成 Python-integer bitsets；candidate rail 的 member、overlap
與 incompatibility 也維持 bit masks。阻擋 feature 則以 coordinate range mask 查詢。
因此一次性的 relation construction 是 `O(F^2)`，而每次 candidate 不再線性掃描
rail 的所有 members。這個取捨針對 geometry 少於約千級、但最終 mesh 可達千萬級
的使用情境。

### 6.3 Sliding active window

features 依 coordinate 遞增處理。若：

```python
feature["coord"] - rail["max_coord"] > merge_tol
```

則目前與後續 features 都不可能再加入該 rail。演算法以 `active_start` 排除這些 rails，減少不必要比較。

## 7. Rail merge safety rules

一個 feature 要加入既有 rail，必須同時通過 rail 內衝突與相鄰 rail topology 檢查。

### 7.1 Coordinate range

合併後：

```python
new_max_coord - new_min_coord <= merge_tol
```

注意這是整條 rail members 的總座標範圍，不只是 feature 到目前 midpoint 的距離。

### 7.2 Active Z overlap

兩個 features 的 active Z interval overlap/touch 判斷概念為：

```python
max(a.z_bottom, b.z_bottom) <= min(a.z_top, b.z_top)
```

不同 Z interval 的 features 有較高機會共用 rail，因為它們不會在同一 extrusion interval 同時要求不同 target coordinates。

### 7.3 XY span overlap/touch

```python
max(a.span_min, b.span_min) <= min(a.span_max, b.span_max)
```

active Z overlap 且 XY span overlap/touch 的不同 coordinates 不能共用 rail，否則同一 node 可能收到互相衝突的 snap targets。

### 7.4 Near endpoint conflict

即使 spans 沒有接觸，若兩條 active-Z-overlapping features 的端點 gap 很小，合併仍可能破壞 corner 附近的 topology。

概念條件：

```python
0 < span_gap <= merge_tol
```

這類 features 會保持在不同 rails。唯一例外是兩個 facing endpoints 都已是
反向 axis 的 pinned structural coordinates，因為它們不會被 rail sharing 拉動。

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

rail 內合法不代表整體 topology 一定合法。加入 feature 時只需檢查可能因該
candidate coordinate 改變的相鄰 rails 與最近 pinned coordinates，確認以下狀態
都不會反序：

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

### 7.7 Pinned coordinates 與 global fallback

root domain boundaries 與無法由 perpendicular coupled snap 完整表示的 feature
endpoints 是 pinned structural coordinates。Pinned feature 只能留在完全相同的 rail
coordinate；該座標也會作為虛擬相鄰 rail 參與 global order 檢查。

由於 greedy 過程後面新建的 rail 可能使早先的 merge 變得不安全，axis
grouping 最後還會重新檢查全部相鄰 rails。若 global order 不成立，該 axis
直接回退為依 exact target coordinate 分組，不會保留有疑慮的 merge。

### 7.8 Geometry-scale snap-plan preflight

rail-local 規則之後，`validate_snap_plan()` 會把 X/Y coupling 作為完整 2D
state 檢查。它列舉每個 exact Z boundary 與相鄰 boundaries 之間的 open
interval；no-op rule 不擴張 touched spans，同 target/span 的重疊 lifecycles 先做
interval union，並對相同 effective rule geometry 只檢查一次。之後驗證：

- 同一 structural node/axis 不會收到不同 targets；
- rule rail、span endpoints 與 target coordinates 都在 structural axes 上；
- coupled X/Y corners 使用同一個同步 state；
- 所有受影響 cells 仍是嚴格凸、正面積 quadrilaterals。

檢查只 materialize rules 觸及的節點與相鄰 cells，成本綁定在幾何與一維
axes，不會配置完整 2D mesh。`_set_pattern()` 先對 rails 與 pinned stations
預演；BOX generation 前再以完整 densified axes（包含 filler stations）預演。
任一 optimized plan 失敗都會重建 `merge_tol=0` 的 exact-coordinate rails，並將
原因寫入 `mesher.rail_optimization_fallback`。Exact plan 仍失敗才是不可恢復
的 error。

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

### 8.5 Pattern plan integrity

`get_snap_rules()` 與 `get_restore_rules()` 會 deep-copy 完整 mapping 或單一 Z 的
rule list，讓 inspection code 可安全排序、標註或修改回傳資料。內部的 `faces`、
`rails`、`snap_rules_by_z`、`restore_rules_by_z`，以及執行所依賴的 axes/pinned
stations 則共同形成 geometry-scale integrity signature。

pattern preprocessing 完成後，這些 execution inputs 是 read-only by contract。
若 caller 繞過 getters 直接修改其中任何部分，下一個 integrity boundary 會
fail closed，而不是以 rules、rails 與 indexes 彼此不一致的狀態執行。合法更新
方式是重新呼叫 `set_pattern_obj()`／`_set_pattern()`，重新建立並 preflight
整份 plan。

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

目前只有 BOX dispatcher branch 能產生可驗證的 mesh。其他 footprint 不可藉由
custom assignment 取代 partition proof，詳見第 3.4 與 9.4 節。

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

span endpoints 與 domain boundaries 中必須維持不動的 coordinates 也會作為
mandatory/pinned stations 加入 axes，防止 densification 產生的大格直接跨過
feature endpoint。

### 9.3 Mesh arrays

```text
nodes     (n_nodes, 3), float64
elements  (n_elements, 4), int32
```

nodes 以 X varying fastest 的 row-major structured grid 排列。
同時 generator 回傳 compact internal metadata：

```python
{
    "kind": "STRUCTURED_BOX",
    "grid_shape": (len(y_nodes), len(x_nodes)),
    "x_nodes": x_nodes,
    "y_nodes": y_nodes,
    "mandatory_axis_coordinates": {...},
    "pinned_axis_coordinates": {...},
}
```

mesher 用這份 metadata 直接做一維 axis lookup 與 node-id arithmetic，不需
掃描/排序全部 elements 來建 rail membership，也不需 custom-mesh 的
node-to-element CSR。metadata 只在生成與驗證成功後安裝，不是 caller 可
任意宣告的 topology shortcut。

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

允許外部 mesher 提供 quadrilateral mesh。通用 assignment path 使用 exact coordinates 檢查
每條 feature span 有連續、共線的 mesh-edge chain，並檢查 mandatory stations
不穿過 quadrilateral 內部。它還會以 bounded chunks 檢查所有 quads，
實際套用每個拓撲不同的 Z state，並在完成後回到 baseline。任一失敗
會原子性恢復 caller attributes 與先前 mesher state。

custom mesh 若確實是 row-major BOX grid，可在 `mesh2d.metadata` 放入
`STRUCTURED_BOX` mapping，或明確傳給
`mesh_assignment(mesh2d, structured_metadata=metadata)`。mesher 會以 bounded chunks exact
比對 axes、所有 node coordinates 與 canonical connectivity，只在通過後套用與內建
generator 相同的 direct arithmetic。未提供 metadata 的 custom mesh 才會建立
通用 rail indexes 與 node-to-element CSR adjacency。對千萬級 2D custom mesh，這些
常駐 arrays 與一次性 coverage/orientation 掃描必須納入 peak RSS。
提供了不一致的 metadata 時 assignment 直接失敗，不會靜默改走通用路徑。

對 explicit BOX domain，通用 path 還會呼叫 `validate_box_domain_partition()`：

1. 檢查所有 referenced nodes 位於 bounds 內。
2. 將每條 undirected edge 編碼並要求 manifold multiplicity 為 1 或 2。
3. 要求每條 multiplicity-one edge 在 physical BOX boundary，並以 exact closed
   interval union 連續覆蓋四邊。
4. 以 `longdouble` 累計正向 quad area，在只由累計次數決定的誤差界內
   比對完整 BOX 面積。

這個 proof 同時拒絕 holes、disconnected regions、internal unmatched edges、
duplicate/non-manifold elements 與 crossing/overlapping embeddings。Verified structured path
以 exact row-major nodes/connectivity、strict axes 與 domain endpoints 做 compact 等價 proof。

explicit POLYGON/CYLINDER footprint 的 custom partition 目前未實作，因此
`mesh_assignment()` 拋出 `NotImplementedError`。Private raw-face caller 可在沒有 domain 時
指派 mesh 來單獨使用 snap API，但之後不能 `build()`。

### 9.5 Authoritative mesh integrity

2D generation 或 assignment 通過後，mesher 記錄 nodes/elements 的 array
identity、shape、dtype、strides、data pointer，以及 canonical baseline
nodes/connectivity digest。Digest 使用 BLAKE2b 並按固定最大 row count 分塊，
因此 hash 千萬級 arrays 時不會建立另一份完整 contiguous copy。

`mesh2d.elements` 與所有將 rail/node ids 映射到 topology 的 structural index
arrays 會設為 read-only。`mesh2d.nodes` 必須保持 writable，因為 snap engine 會在
同一 authoritative buffer 上做 managed coordinate updates；這些更新會由 baseline
restore 與 active-state bookkeeping 管理。

在 `build()` 使用 indexes 前，mesher 會先確認 public arrays 與 structural arrays
沒有被 replacement，將 managed state 正規化回 baseline，然後重新計算
nodes/connectivity digest 並比對 structural digest。一般 connectivity/index
in-place mutation 會被 read-only flag 直接阻止；若 caller 強制解鎖或直接修改
nodes，digest/layout validation 仍會 fail closed。任何外部 mesh 變更都必須透過
新的 `mesh_assignment()` 重建 indexes，不能沿用舊 topology mapping。

## 10. Rail node index

每條 rail 最後都以下列 logical index 表示：

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

內建 structured BOX mesh 由 X/Y axes 直接產生這些 arrays，無需排序全 mesh。
沒有 structured metadata 的通用 custom mesh 才會由 `elements` 實際引用
的 active nodes 建立並排序。

對單一 snap rule，可使用 binary search 找到 span：

```python
lo = searchsorted(span_values, span_min, side="left")
hi = searchsorted(span_values, span_max, side="right")
node_ids = sorted_node_ids[lo:hi]
```

預設 span selection 是 exact，這避免將極接近但不同的 structural rails
alias 在一起，也避免每個 rule 掃描全部 mesh nodes。

## 11. Stage F：套用 active snap state

入口：

```python
mesher.apply_snap_rules_at_z(z)
```

流程：

1. 以 exact Z（或 caller 明確提供的解析容差）決定查詢 state。
2. 取得所有滿足 `z_bottom <= z <= z_top` 的 snap rules。
3. 以 structural rail index 計算每條 rule 的 span nodes。
4. 加入需要一起移動的 coupled corner nodes。
5. 檢查同一 node/axis 是否有相互衝突的 targets。
6. 將上一 managed state 改動過的 nodes 恢復為 baseline，再寫入目前 targets。
7. 只驗證改動 nodes 相鄰的 quadrilaterals；若變成逆向、退化、凹或
   self-intersecting，整批 coordinate updates 回滾。

### 11.1 為什麼 batch apply

同一 Z 可能有多條 X/Y perpendicular snaps，也可能與上一次查詢是完全
不同的 active state。

程式先計算所有 node selections，再統一更新 coordinates，避免 rule 執行順序改變後續 node lookup 結果。

### 11.2 Baseline restore 與 node-buffer ownership

恢復不來自 restore rule queue。mesher 記錄上一 managed state 在 X/Y 各自改動
的 sparse node ids，下一次 transaction 先根據 structural rail id 回寫該 axis 的
baseline coordinate。這份 sparse state 綁定到一個具體 node buffer；如果尚有
active changes 就嘗試切換到另一個 array，mesher 會 fail closed，避免將舊 node ids
恢復到錯誤資料。

### 11.3 Coupled corners

X rule 與 Y rule 若 target spans 相交，交點 node 可能需要同時更新 X 與 Y。

判斷概念：

```python
x_rule.span contains y_rule.target_coord
and
y_rule.span contains x_rule.target_coord
```

符合時，corner node 會加入兩條 rules 的 affected node set。

### 11.4 Inclusive top Z 與 order-independent state

假設 feature：

```text
z_bottom = 0
z_top    = 10
```

事件：

```text
apply z=0   -> snap
apply z=10  -> feature 仍 active，維持真實 coordinate
apply z>10  -> feature 不再 active，同一 transaction 回到 rail baseline
```

可以直接從任一 Z 跳到另一 Z，結果只由新 Z 的 active rules 決定。
若 caller 手動改過 rail coordinates，或要明確回到完整 baseline，可呼叫：

```python
mesher.reset_snap_state()
```

`apply_snap_rules_at_z()` 的 `eps` 預設為 `0.0`，不會對 event rounding。明確
提供正容差時，它只把查詢 Z 解析到唯一相鄰 event；同時等距於兩個
events 會被拒絕，且查詢容差不會用在 XY topology matching。

## 12. Stage G：3D organize 與 drag

入口：

```python
dragger = mesher.build(
    obj_list,
    preserve_mesh2d=False,
    allow_independent_bodies=False,
)
```

`build()` 要求 explicit BOX domain。沒有 domain 時拋出 `RuntimeError`；非 BOX
domain 則因 partition proof 尚未實作而拋出 `NotImplementedError`。

預設只允許一個至少有兩筆 entries 的 object stack。每個 stack 會建立各自
的 3D nodes，接觸介面也不共點；多 stack 所代表的是 non-conformal independent
bodies。只有 caller 明確傳入 `allow_independent_bodies=True` 才允許。
`allow_independent_bodies` 與 `preserve_mesh2d` 都要求真正的 boolean 值；字串或
整數不會被 truthiness 隱式接受，以免意外改變 FEM topology 或 state ownership。

### 12.1 Build preflight 與 state ownership

修改 mesh 或產生任何 3D output 前，`build()` 會：

1. 檢查每個 stack 的 Z 為有限值且嚴格遞增。
2. 要求 stack 範圍內的所有 exact pattern `z_bottom`/`z_top` 都是
   layer boundary。
3. 驗證每個非 sentinel layer 都有 `areas` 與正值有限 `element_size`。
4. 驗證 area/metal material schema，不允許未知 type 或非法 density 靜默退回
   base material。
5. 對每個 slab 的 area outer boundaries、area holes、metal ranges 與 metal
   holes 證明 exact representability。

第 5 步將每條 BOX/POLYGON selector edge 轉成 horizontal/vertical spans，再要求
它由下列 intervals 在整個 closed slab `[z_begin, z_end]` 連續覆蓋：

- active Z 完整包含該 slab 的 original pattern edges；
- 永久 BOX domain edges；
- 在該 slab 中沒有被任何 snap rule 位移的 structural rail portions。

線段 union、coordinate 與 Z coverage 皆為 exact。某條邊只在 slab 的部分高度存在，
或 baseline rail 在期間曾被拉走，都會拒絕，避免 whole-element selection 靜默
漏材或越界。所有 spans 是 closed intervals：位移 span 的 endpoints 同樣是被拉動的
structural nodes，不屬於 baseline 可用範圍。實作以 `nextafter(endpoint, +/-inf)`
保留兩側可表示部分，故 selector 不能把 displaced endpoint 當合法 baseline
boundary；只有其他 active pattern/domain edge exact 覆蓋時才可補足。CYLINDER
selector 不是 orthogonal exact boundary，目前拒絕。

preflight 通過後且在 3D extrusion 前，`build()` 還會執行第 9.5 節的 authoritative
mesh check：先驗證 array/index identity 與 layout，再用 bounded chunks 重算
baseline XY nodes 和 quadrilateral connectivity digest。這個線性安全掃描每次 build
執行一次，不隨 Z subdivisions 重複。

`preserve_mesh2d=True` 時，build 在唯一一份 working-node copy 上從 baseline
開始。如果原 2D buffer 進入 build 前已處於 active snap state，它的 exact
coordinates、sparse changed-node ids 與 buffer ownership 在 build 成功或失敗後都會
恢復為原狀，而不是強制改回 baseline。

Area `holes`、metal `ranges`／`holes` 都只接受 list/tuple selector collections。
一次性 iterator/generator 會在 preflight 拒絕，確保 validation 與 runtime
classification 使用完全相同的 geometry。

每個 Z interval：

1. `apply_snap_rules_at_z(z_begin)`。
2. 若已有上一層 3D top nodes，同步被 snap 的 XY。
3. 重新計算 2D element areas。
4. `_organize(areas)` 將本層 material patches overlay 到上一個 interval。
5. `_drag(element_size, z_begin, z_end)` 建立 3D nodes 與 hexahedra。

同一 object stack 的 material state 會跨 interval 保留。第一層從 `EMPTY` 開始；
後續 layer 只有 area 命中的 elements 被重新分配，沒有命中的 elements 繼承前一層。
`previous_element_2D_comp` 在一次 organize 中保持為上一層的固定快照，供
`CONTINUE`／`CONVERT` 判斷；新 current buffer 只複製 previous active footprint，
不掃描或複製未使用的 2D cells。另一個 object stack 開始時才清空狀態。

所有 intervals 完成後，final sentinel 的 complete active snap state 仍會套用到 2D
working mesh，並同步到已建立的最終 3D top plane。Sentinel 不產生新
interval，但絕對不會被當成可忽略的幾何 state。

`Obj` hierarchy 不會自動轉換成這份 3D material stack。

### 12.2 Area selection

每個 2D quadrilateral 以四個 corners 進行幾何判斷。四點都在 area 內時才選入。

- BOX：比較 element min/max bounds。
- POLYGON：四點都需位於任一 hull 且不在 holes。

polygon boundary 使用 inclusive test，位於邊界上的 corners 視為 inside。

Area 是目前 layer 的 overlay patch，而不是完整 cross-section。Area hole 代表該
patch 不覆寫 hole 內 elements；若該處已在上一層 active，其 material 仍會繼承。

搜尋以 bounded chunks 建立單次 element-coordinate work set，不會將整個 2D mesh
複製為 corner-coordinate array。同一 layer 中的 areas 不得對同一 element 重疊
主張所有權；發現 overlap 就拋出 `ValueError`。需要 priority region 時，必須
以 explicit holes 表達，而不依賴 area list ordering。

當 mesh 與 selector 數量都超過門檻時，`_organize()` 會針對當下 node coordinates
建立一次 temporary uniform-bin index。索引 key 使用 element connectivity 的真實
corner，不使用可能有浮點平均誤差的 centroid。BOX/POLYGON bbox 只作保守候選
縮減；每個候選仍必須通過完全相同的四角與 hole exact predicate，任何不支援或
不安全的 selector/subset 形式都退回 bounded full scan。索引在 `_organize()` 結束
立即丟棄，不跨越後續 snap mutation。

### 12.3 Material assignment

`Dragger.comps`：

```python
{"EMPTY": 0}
```

新材料在第一次出現時取得新的整數 id。

在 `_organize()` 執行前，high-level `build()` 已對所有 material fields 做完整
schema preflight：

- 每個 area 的 `material` 與每個 metal 的 `material` 必須是 non-empty string。
- `CONVERT` 的 `material_o` 也必須是 non-empty string。
- 上述 labels 都不可等於 reserved component `EMPTY`（id `0`）；排除區域必須用
  explicit holes 表達。Hole 不會清除前一層既有 material，只會排除本次覆寫。
- metal `type` 是 case-sensitive，只能 exact 等於 `NORMAL`、`CONTINUE` 或
  `CONVERT`。
- `NORMAL.density` 必須存在、是 numeric、不是 bool、為 finite，且落在
  inclusive `[0, 100]`。

因此 unknown/lowercase type、missing/empty labels、`NaN`、`inf` 或越界 density
會在任何 mesh mutation 與 3D allocation 前拋出 `ValueError`。低階 organizer
不會再因為未識別 metal 而把元素留給 area base material，避免 schema error
變成看似成功但材料錯誤的 mesh。

Area metals 的處理順序：

1. 計算 `NORMAL` metal 的候選 volume。
2. 處理 `CONTINUE`。
3. 處理 `CONVERT`。
4. 處理 `NORMAL` density assignment。
5. 剩餘 elements 指派 area material。

每個 selector 只在自己的 priority pass 計算一次並立即消費；NORMAL 的候選 volume
與 assignment 共用該結果。Remaining pool 使用一份 area-local boolean mask，避免
每個 metal 反覆排序或掃描完整 pool，也不保留 `selectors × elements` 的 hit arrays。
若 individual areas 雖有限但 selector sum 超出 float64 range，立即 fail closed；
不允許 `inf` 進入 density target/cumulative comparison。

### 12.4 Z subdivision

對距離：

```python
distance = end - begin
```

drag subdivision：

```python
drag_num = max(1, ceil(distance / preferred_element_size))
actual_size = distance / drag_num
```

此處使用的是 Z direction element size。
它是最大允許尺寸，不只是 preference。

在當前 slab 的 node mapping 或 output allocation 之前，`_validate_drag_z_planes()`
以 chunks 證明：

- 所有 float64 planes 有限、嚴格遞增且不重複；
- final plane exact 等於 `end`；
- interval sizing 在 requested maximum 與最多數個 representable steps 的算術
  allowance 內，但 topology order 無容差；
- nodes/elements 不超過 int32 capacity。

當 `element_size` 小於該絕對 Z 的 float64 resolution、subdivision 不可表示，
或 planes 會重複/反轉時，立即 fail closed，不會產生 zero-volume hexes。
Simple full-domain exact-capacity path 也會先對全部 slabs 執行同一 proof，確認後才
一次配置 final arrays。

### 12.5 Hexahedral connectivity

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
7. Z events 不 rounding；feature 在 `z_top` 仍 active，在任一更高 Z 不再
   active，並由完整 state transaction 恢復 baseline。
8. X/Y 同 Z corner snaps 必須同步處理。
9. Optimized snap plan 在結構 axes 與完整 densified axes 都必須通過
   preflight；否則必須改用 exact-coordinate rails。
10. 座標 topology equality 是 exact；相差 1 ULP 的 rails/targets 不得 alias。
11. Custom mesh 的每個 rule span 必須是連續 exact mesh-edge chain，mandatory
    stations 不得穿過 element 內部。
12. Explicit BOX domain 的 custom mesh 必須是 inside、manifold、無 holes/overlap
    且完整覆蓋 physical boundary 的 partition。
13. `build()` 不得跨過沒有 exact layer boundary 的 pattern Z event，且每個
    area/range/hole boundary 必須在整個 slab exact 可表示。
14. Final sentinel 的 snap state 必須同步到最終 3D top plane。
15. 多 object stacks 預設拒絕；明確 opt-in 時它們是各自擁有 nodes 的
    non-conformal bodies。
16. `preserve_mesh2d=True` 必須保留進入 build 前的 node coordinates 與
    managed snap-state ownership，包含原本已 active 的狀態。
17. Z planes 必須可用 float64 表示為嚴格遞增序列；否則 fail closed。
18. 同一 layer 的 areas 不得重疊主張同一 element。
19. `Dragger` 預先配置 arrays，只能以 `node_num`／`element_num` 判斷有效範圍。
20. Rule getters 必須回傳 deep copies；validated faces/rails/rules 被直接修改時
    必須 fail closed。
21. Authoritative mesh array replacement 或未受管理的 in-place mutation 不得沿用
    舊 structural indexes；connectivity/index arrays 維持 read-only。
22. Baseline rail 可用 span 採 closed-interval subtraction，displaced endpoints
    不得作為合法 baseline selector boundary。
23. High-level build 的 area/metal labels、exact metal type 與 `NORMAL` density
    必須在 geometry preflight 驗證；非法 schema 不得回退為 base material。

## 14. 複雜度概觀

令：

- `F`：feature line 數量；
- `R`：shared rail 數量；
- `N`：2D node 數量；
- `Q_z`：某一 Z event 的 rules 數量。

大致成本：

```text
feature sorting          O(F log F)
rail relation precompute O(F^2), compact bitsets
rail grouping            依 active candidate rails 與 bit operations 數量而定
generated rail index     O(R + axis lengths), direct structured arithmetic
custom rail index        O(N log N) 級別的排序工作
custom coverage/index    O(N log N + number_of_quads), bounded/chunked scans
BOX partition proof      O(E log E), retaining four compact edge keys per quad
build face search        worst-case O(E per selector pass); large cases use
                         temporary corner-bin candidates + exact predicates
single Z snap            O(Q_z log nodes_per_rail + touched_nodes)
2D mesh generation       O(N + number_of_quads)
3D drag                  O(generated_nodes + generated_hexes)
```

實際效能收益主要來自：

- shared rails 減少必要 XY grid lines；
- active rail window 減少 grouping candidates；
- generated structured metadata 避免 full-mesh rail/adjacency preprocessing；
- custom sorted rail node index 加速 span queries；
- NumPy vectorization 與 bounded flat-output chunks 批次建立 3D nodes/elements；
- simple full-domain BOX stacks 在 extrusion 前精確預留 final capacity，避免 growth copy。

geometry 小於千級時，rail grouping 與 snap-plan preflight 仍與幾何、axes 與受影響
cells 同階，不會與最終千萬級 hexahedra 同階。大型資料的主要成本是 final
arrays、base 2D mesh 與 custom-mesh indexes；詳見 [Performance and capacity
guidance](performance.md)。

## 15. Extension points

### 15.1 外部 2D mesher

流程：

```python
mesher.set_pattern_obj(...)
snap_faces = mesher.get_snap_faces()
custom_mesh = external_mesher(snap_faces)
mesher.mesh_assignment(custom_mesh)
```

若 external mesher 產生 exact row-major BOX grid，可以附上第 9.4 節的
`STRUCTURED_BOX` metadata，以驗證後的 direct indexing 取代通用 CSR 路徑。

### 15.2 CYLINDER symbol 不是已開放的 extension point

repository 仍保留下列 low-level symbol：

```python
checkerboard_mesh_cylinder(
    domain,
    x_list,
    y_list,
    element_size,
)
```

但只補上 generator 或提供外部 `Mesh2D` 不足以開放 high-level pipeline。
`mesh_assignment()` 必須能證明整個 footprint 無 holes/overlap/non-manifold topology，
`build()` 也必須證明所有 slab selector boundaries。這些 CYLINDER proof 尚未實作，
所以兩個入口都 fail closed。

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
| custom BOX domain partition / non-BOX fail closed | `tests/test_mesh_assignment.py` |
| exact custom feature/station coverage | `tests/test_mesh_coverage.py` |
| snap-plan safety preflight/fallback | `tests/test_snap_plan_validation.py` |
| rule getter copies / pattern-plan mutation rejection | `tests/test_get_snap_rules.py` |
| active-state snap/restore execution | `tests/test_apply_snap_rules_at_z.py` |
| 10M-scale complexity/allocation invariants | `tests/test_performance_regression.py` |
| 2D area/material drag engine | `tests/test_drag_engine.py` |
| area overlap rejection / chunked face search | `tests/test_drag_engine.py` |
| slab boundary, closed-span endpoints, final sentinel, preserved state, multi-body policy | `tests/test_optimal_mesh_build.py` |
| mesh array/index integrity and read-only connectivity | `tests/test_optimal_mesh_build.py` |
| high-level material schema fail-closed validation | `tests/test_optimal_mesh_build.py` |
| full `OptimalMesh25D.build()` | `tests/test_optimal_mesh_build.py` |

修改 rail compatibility 時，至少應覆蓋：

- same-Z span overlap；
- one-ULP distinct topology coordinates；
- same-Z span separation；
- cross-Z sharing；
- near endpoint conflict；
- blocking intermediate feature；
- adjacent rail order；
- exact Z event 與 order-independent active-state restore；
- unsafe optimized-plan exact-rail fallback；
- disconnected/missing custom mesh edge-chain coverage；
- BOX holes/non-manifold/overlap partition rejection；
- whole-slab build selector boundary coverage；
- final sentinel state synchronization；
- active-state preservation 與 explicit independent-body opt-in；
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

Z=10 呼叫時，features 仍為 active，所以兩段 X rail 保持真實 targets。
查詢任一 `Z > 10` 的 state 時，這些 rules 已不 active，transaction 會先將
曾改動的 nodes 還原到 `x=1.25`。

這個例子展示此專案的核心：以較少的 structural rails 表示多個分層 pattern coordinates，再以 Z-aware rules 恢復真實幾何。
